"""Small durable state shared by concurrent executor processes: journal, slot health, billed ledger.

An agent may fire many executors at once, so every write takes an exclusive file lock and is a
single append or an atomic replace. Readers tolerate a torn last line.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

STATE = Path(os.environ.get("PIVOT_STATE") or Path.home() / ".cache" / "jev-pivot-lab" / "state")


@contextmanager
def locked(path: Path):
    """Exclusive lock on `<path>.lock`, held for the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with locked(path), open(path, "a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:      # a line another process is still writing
            continue
    return rows


class Health:
    """Which slots are failing right now, so a swarm does not keep hammering a dead one.

    A slot that errors (not a wrong answer: a rate limit, a login that lapsed, an outage) cools
    down for 30 s, doubling per consecutive error up to 10 minutes. One success clears it.
    """

    BASE_S, MAX_S = 30.0, 600.0

    def __init__(self, path: Path | None = None, clock=time.time) -> None:
        self.path, self.clock = Path(path or STATE / "health.json"), clock

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def blocked(self, slot_id: str) -> str | None:
        entry = self._load().get(slot_id)
        if entry and entry.get("until", 0) > self.clock():
            return f"cooling down for {entry['until'] - self.clock():.0f}s after: {entry.get('error', 'error')}"
        return None

    def record(self, slot_id: str, ok: bool, error: str | None = None) -> None:
        with locked(self.path):
            data = self._load()
            if ok:
                if slot_id not in data:
                    return
                data.pop(slot_id)
            else:
                errors = int(data.get(slot_id, {}).get("errors", 0)) + 1
                data[slot_id] = {"errors": errors, "error": str(error or "error")[:200],
                                 "until": self.clock() + min(self.MAX_S, self.BASE_S * 2 ** (errors - 1))}
            partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            partial.write_text(json.dumps(data), encoding="utf-8")
            partial.replace(self.path)

    def snapshot(self) -> dict:
        now = self.clock()
        return {slot: entry for slot, entry in self._load().items() if entry.get("until", 0) > now}


class BilledLedger:
    """A per-day cap on billed gateway spend across every executor process on this machine.

    A call reserves its worst-case cost before it is sent and settles to what it cost after, so
    concurrent processes cannot together pass the cap. A reservation that is never settled (a
    killed process) stops counting after 15 minutes.
    """

    STALE_S = 900.0

    def __init__(self, path: Path | None = None, clock=time.time) -> None:
        self.path, self.clock = Path(path or STATE / "billed.jsonl"), clock

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    def committed(self, provider: str | None = None) -> float:
        day, now = self._today(), self.clock()
        open_, total = {}, 0.0
        for row in read_jsonl(self.path):
            if row.get("day") != day or (provider and row.get("provider") != provider):
                continue
            if row.get("kind") == "reserve":
                if now - row.get("ts", 0) < self.STALE_S:
                    open_[row["id"]] = float(row.get("usd", 0.0))
            elif row.get("kind") == "settle":
                open_.pop(row.get("id"), None)
                total += float(row.get("usd", 0.0))
        return total + sum(open_.values())

    def reserve(self, provider: str, usd: float, cap: float) -> str | None:
        """-> a reservation id, or None when the day's cap would be passed."""
        record = {"kind": "reserve", "id": uuid.uuid4().hex, "provider": provider, "usd": round(usd, 6),
                  "day": self._today(), "ts": self.clock(), "pid": os.getpid()}
        line = json.dumps(record) + "\n"
        with locked(self.path):
            if self.committed() + usd > cap:
                return None
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(line)
        return record["id"]

    def settle(self, reservation: str, provider: str, usd: float) -> None:
        append_jsonl(self.path, {"kind": "settle", "id": reservation, "provider": provider,
                                 "usd": round(usd, 6), "day": self._today(), "ts": self.clock()})
