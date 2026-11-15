"""Run every slot on every item once, behind an append-only cache and hard spend caps.

The grid is the expensive part and is paid for once: routing policies are replayed against it
afterwards for the price of a few Jev calls.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import state
from .slots import SYSTEM, Slot, run_slot, usd, worst_case_usd
from .tasks import grade


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class Cache:
    """Receipts keyed by (slot, system prompt, prompt). A paid call is never repeated."""

    def __init__(self, path: Path) -> None:
        self.path, self.rows, self.lock = Path(path), {}, threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for row in state.read_jsonl(self.path):
            self.rows[row["key"]] = row["receipt"]

    @staticmethod
    def key(slot_id: str, system: str, prompt: str) -> str:
        return _digest([slot_id, system, prompt])

    def get(self, key: str, timeout: float) -> dict | None:
        receipt = self.rows.get(key)
        # A cached timeout only stands if it was given at least as long as this run allows.
        if receipt and receipt.get("error") == "timeout" and receipt.get("timeout_s", 0) < timeout:
            return None
        return receipt

    def put(self, key: str, receipt: dict) -> None:
        with self.lock:
            self.rows[key] = receipt
            state.append_jsonl(self.path, {"key": key, "receipt": receipt})      # safe across processes too


class Budget:
    """Stops launching new calls once a provider's spend or the call count hits its cap.

    For the subscription CLIs the figure is API-equivalent and a call in flight may run a little
    past the cap. A metered call passes `reserve`, its worst-case cost: it is only sent if the cap
    holds even when every call in flight costs its worst case, so billed spend cannot pass the cap.
    """

    def __init__(self, max_usd: dict[str, float], max_calls: int) -> None:
        self.max_usd, self.max_calls = dict(max_usd), max_calls
        self.spent = {provider: 0.0 for provider in max_usd}
        self.reserved: dict[str, float] = {}
        self.calls, self.lock = 0, threading.Lock()

    def allow(self, provider: str, reserve: float = 0.0) -> bool:
        with self.lock:
            cap = self.max_usd.get(provider, float("inf"))
            committed = self.spent.get(provider, 0.0) + self.reserved.get(provider, 0.0)
            if self.calls >= self.max_calls or committed >= cap or (reserve and committed + reserve > cap):
                return False
            self.calls += 1
            self.reserved[provider] = self.reserved.get(provider, 0.0) + reserve
            return True

    def add(self, provider: str, dollars: float | None, reserve: float = 0.0) -> None:
        with self.lock:
            self.reserved[provider] = max(0.0, self.reserved.get(provider, 0.0) - reserve)
            self.spent[provider] = self.spent.get(provider, 0.0) + (dollars or 0.0)

    def summary(self) -> dict:
        return {"calls": self.calls, "max_calls": self.max_calls,
                "usd_equivalent": {p: round(v, 4) for p, v in self.spent.items()}, "max_usd": self.max_usd}


class Breaker:
    """Stops calling what keeps failing instead of burning calls on it.

    A slot that errors twice running without ever answering is a model this login does not serve.
    A provider whose working slots error many times running has hit a rate limit: stop, and let
    a later run pick up from the cache.
    """

    def __init__(self, slot_limit: int = 2, provider_limit: int = 8) -> None:
        self.slot_limit, self.provider_limit = slot_limit, provider_limit
        self.slot_errors: dict[str, int] = {}
        self.provider_errors: dict[str, int] = {}
        self.worked: set[str] = set()
        self.reasons: dict[str, str] = {}
        self.lock = threading.Lock()

    def blocked(self, slot: Slot) -> str | None:
        with self.lock:
            if self.provider_errors.get(slot.provider, 0) >= self.provider_limit:
                return f"{slot.provider} stopped after {self.provider_limit} errors in a row"
            if self.slot_errors.get(slot.id, 0) >= self.slot_limit:
                return f"slot stopped: {self.reasons.get(slot.id, 'error')}"
            return None

    def record(self, slot: Slot, receipt: dict) -> None:
        with self.lock:
            if receipt.get("ok") or receipt.get("error") == "timeout":
                self.worked.add(slot.id)
                self.slot_errors[slot.id] = self.provider_errors[slot.provider] = 0
                return
            self.slot_errors[slot.id] = self.slot_errors.get(slot.id, 0) + 1
            self.reasons[slot.id] = str(receipt.get("error"))[:200]
            if slot.id in self.worked:      # a slot that was answering now is not: the provider's doing
                self.provider_errors[slot.provider] = self.provider_errors.get(slot.provider, 0) + 1

    def summary(self) -> dict:
        return {"stopped_slots": {slot: self.reasons.get(slot, "") for slot, n in sorted(self.slot_errors.items())
                                  if n >= self.slot_limit},
                "stopped_providers": sorted(p for p, n in self.provider_errors.items() if n >= self.provider_limit)}


def execution_order(items: list[dict]) -> list[dict]:
    """Round-robin across tiers so a grid cut short by the budget still spans every difficulty."""
    seen: dict[int, int] = {}
    keyed = []
    for item in items:
        seen[item["tier"]] = seen.get(item["tier"], 0) + 1
        keyed.append((seen[item["tier"]], item["tier"], item))
    return [item for _, _, item in sorted(keyed, key=lambda entry: entry[:2])]


def run_cell(item: dict, slot: Slot, *, cache: Cache, budget: Budget, prices: dict, timeout: float,
             system: str = SYSTEM, runner=run_slot, breaker: Breaker | None = None) -> dict:
    key = Cache.key(slot.id, system, item["prompt"])
    receipt = cache.get(key, timeout)
    cached = receipt is not None
    if receipt is None:
        skipped = {"item": item["id"], "slot": slot.id, "status": "skipped", "correct": False, "cached": False}
        reason = breaker.blocked(slot) if breaker else None
        if reason:
            return {**skipped, "reason": reason}
        reserve = worst_case_usd(slot, system, item["prompt"])
        if not budget.allow(slot.provider, reserve):
            return {**skipped, "reason": "budget cap"}
        receipt = {**runner(slot, item["prompt"], system=system, timeout=timeout), "timeout_s": timeout}
        budget.add(slot.provider, usd(receipt, prices), reserve)
        if receipt["ok"] or receipt["error"] == "timeout":
            cache.put(key, receipt)
    if breaker:
        breaker.record(slot, receipt)
    status = "ok" if receipt["ok"] else ("timeout" if receipt["error"] == "timeout" else "error")
    return {"item": item["id"], "slot": slot.id, "status": status, "cached": cached,
            "correct": bool(receipt["ok"] and grade(item, receipt["text"])),
            "usd": usd(receipt, prices), **receipt}


def run_grid(items: list[dict], slots: list[Slot], *, cache: Cache, budget: Budget, prices: dict,
             timeout: float = 300.0, concurrency: int = 4, system: str = SYSTEM, runner=run_slot,
             breaker: Breaker | None = None, log=print) -> list[dict]:
    cells = [(item, slot) for item in execution_order(items)
             for slot in sorted(slots, key=lambda s: (s.rank, s.provider))]
    done = [0]

    def work(cell):
        item, slot = cell
        row = run_cell(item, slot, cache=cache, budget=budget, prices=prices, timeout=timeout,
                       system=system, runner=runner, breaker=breaker)
        done[0] += 1
        mark = {"ok": "pass" if row["correct"] else "FAIL"}.get(row["status"], row["status"].upper())
        extra = f" {row.get('reason', '')}" if row["status"] == "skipped" else (
            f" {row['wall_s']:6.1f}s out={row['usage']['output']:<6} "
            f"${row['usd'] or 0:.4f}{' cached' if row['cached'] else ''}"
            f"{'  ' + str(row.get('error'))[:120] if row['status'] == 'error' else ''}")
        log(f"[grid {done[0]:>4}/{len(cells)}] {item['id']:<22} {slot.id:<32} {mark:<7}{extra}")
        return row

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        rows = list(pool.map(work, cells))
    position = {(item["id"], slot.id): n for n, (item, slot) in
                enumerate((item, slot) for item in items for slot in slots)}
    return sorted(rows, key=lambda row: position[(row["item"], row["slot"])])
