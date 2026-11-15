"""The compiled routing policy: everything the executor needs from a measured grid, in one file.

`pivot fit` turns a grid run into a policy: for each Jev difficulty bucket the efficient slots
with their measured records, the reference slot (the cheapest one that was right most often),
and the order to fall back in when a provider is down. The executor only ever reads this file;
it never touches a grid, so a run cannot be broken by a half-finished experiment.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import time
from pathlib import Path

from . import meta
from .router import BUCKETS, DIFFICULTY, bucket_best, bucket_of, jev_state, profile_questions, slot_stats
from .slots import Slot

SCHEMA = "jev-pivot.policy/v1"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "policy" / "default.json"
OPTION_KEYS = ("slot", "solved", "n", "p", "cost", "wall", "price", "overall_solved", "overall_n")


class PolicyError(ValueError):
    """The policy file is missing, malformed, or names a slot it does not define."""


def compile_policy(grid, rows: list[dict], jev, *, source: str, time_value: float = 0.0) -> dict:
    """Fit on the whole grid. `rows` are the grid's cells, which carry each slot's provider and price."""
    buckets = {item["id"]: bucket_of(jev.ask(jev_state(item), profile_questions())) for item in grid.items}
    stats = slot_stats(grid, grid.items)
    reliable = bucket_best(grid, stats, grid.items, buckets)["reliable"]
    reference = reliable[0]
    card = meta.cards(grid, grid.items, buckets)
    options = {bucket: meta.options(card, bucket, reference, time_value) for bucket in BUCKETS}
    named = set(reliable) | {option["slot"] for opts in options.values() for option in opts}
    defined = {}
    for row in rows:
        if row["slot"] in named and row["slot"] not in defined and row.get("provider"):
            price = row.get("price")
            defined[row["slot"]] = {
                "provider": row["provider"], "model": row["model"], "effort": row["effort"],
                "price": [price[key] for key in ("input", "cached_input", "output")] if price else None,
                "solved": stats[row["slot"]]["solved"], "n": stats[row["slot"]]["n"],
                "cost": stats[row["slot"]]["cost"],
                "wall": sum(grid.wall(item["id"], row["slot"]) for item in grid.items) / len(grid.items)}
    digest = hashlib.sha256(json.dumps(sorted((row["item"], row["slot"], bool(row.get("correct"))) for row in rows))
                            .encode("utf-8")).hexdigest()
    return {
        "schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"run": source, "problems": len(grid.items), "slots": len(grid.slot_ids),
                   "metric": grid.metric, "time_value": time_value, "grid_sha256": digest},
        "jev_model": sorted(jev.resolved)[0] if getattr(jev, "resolved", None) else None,
        "difficulty": DIFFICULTY, "reference": reference, "reliable": reliable, "slots": defined,
        "buckets": {str(bucket): {
            "problems": sum(b == bucket for b in buckets.values()),
            "options": [{key: option[key] for key in OPTION_KEYS if key in option} for option in options[bucket]],
            # Every reliable slot's record on this bucket, for when the usual reference is not usable
            # (its provider disallowed or down) and another reliable slot has to stand in for it.
            "standby": [{key: option[key] for key in OPTION_KEYS if key in option}
                        for option in meta.options(card, bucket, reference, time_value, keep=reliable)
                        if option["slot"] not in {o["slot"] for o in options[bucket]}]}
            for bucket in BUCKETS}}


class Policy:
    def __init__(self, data: dict, path: Path | None = None) -> None:
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise PolicyError(f"not a {SCHEMA} file")
        try:
            self.slots = {slot_id: Slot(spec["provider"], spec["model"], spec["effort"], 0,
                                        tuple(spec["price"]) if spec.get("price") else None)
                          for slot_id, spec in data["slots"].items()}
            self.records = data["slots"]
            self.reliable = list(data["reliable"])
            self.buckets = {int(bucket): entry["options"] for bucket, entry in data["buckets"].items()}
            self.standby = {int(bucket): {option["slot"]: option for option in entry.get("standby", [])}
                            for bucket, entry in data["buckets"].items()}
            self.source, self.created = data["source"], data["created"]
            self.system = data.get("system")        # the instructions its slots were measured under, if custom
        except (KeyError, TypeError, ValueError) as error:
            raise PolicyError(f"malformed policy: {error!r}") from error
        named = set(self.reliable) | {option["slot"] for options in self.buckets.values() for option in options}
        missing = sorted(named - set(self.slots))
        if missing or not self.reliable or set(self.buckets) != set(BUCKETS):
            raise PolicyError(f"policy is incomplete (undefined slots: {missing[:3]})")
        for slot_id, slot in self.slots.items():
            if slot.id != slot_id:
                raise PolicyError(f"slot {slot_id!r} does not match its definition {slot.id!r}")
        self.data, self.path = data, path

    def age_days(self, now: float | None = None) -> float:
        made = calendar.timegm(time.strptime(self.created, "%Y-%m-%dT%H:%M:%SZ"))
        return ((now or time.time()) - made) / 86400

    def usable(self, slot_id: str, allowed, blocked) -> bool:
        """`allowed(slot)` is the caller's allow-list; `blocked(slot_id)` a reason it is cooling down, or None."""
        return bool(allowed(self.slots[slot_id])) and not blocked(slot_id)

    def reference(self, allowed, blocked) -> str | None:
        """The cheapest fully reliable slot that may be used right now."""
        return next((slot for slot in self.reliable if self.usable(slot, allowed, blocked)), None)

    def options(self, bucket: int, reference: str, allowed, blocked) -> list[dict]:
        """The bucket's efficient options that may be used right now, with the reference always present."""
        kept = [dict(option) for option in self.buckets[bucket] if self.usable(option["slot"], allowed, blocked)]
        if reference not in [option["slot"] for option in kept]:
            record = self.records[reference]
            kept.append(dict(self.standby[bucket].get(reference) or {
                "slot": reference, "solved": 0, "n": 0, "p": record["solved"] / record["n"],
                "cost": record["cost"], "wall": record["wall"],
                "overall_solved": record["solved"], "overall_n": record["n"]}))
        for option in kept:
            option.setdefault("price", option["cost"])
        return sorted(kept, key=lambda option: (option["price"], -option["p"]))

    def fallbacks(self, first: str, allowed, blocked) -> list[str]:
        """Where to go when a slot's provider fails: reliable slots, another provider's first."""
        usable = [slot for slot in self.reliable if slot != first and self.usable(slot, allowed, blocked)]
        home = self.slots[first].provider if first in self.slots else None
        other, seen = [], set()
        for slot in usable:
            provider = self.slots[slot].provider
            if provider != home and provider not in seen:
                other.append(slot)
                seen.add(provider)
        return other + [slot for slot in usable if slot not in other]


def load_policy(path: Path | str | None = None) -> Policy:
    target = Path(path) if path else DEFAULT_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except OSError as error:
        raise PolicyError(f"no policy at {target}; build one with `pivot fit --from <run>`") from error
    except ValueError as error:
        raise PolicyError(f"{target} is not valid JSON") from error
    return Policy(data, target)
