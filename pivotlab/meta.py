"""Derived model selection: measured slot cards + Jev's read of the turn + the budget -> a slot.

Nothing here is a hand-written rule about which model suits which task. A card says what a slot
did on calibration problems of one Jev difficulty bucket (never the problem being routed). The
selection is derived from the cards, either by arithmetic (`derive`) or by Jev itself (`ask`),
which is shown the same cards together with the budget and the conversation.

The target is the reference slot's answer quality (the cheapest single slot that was right most
often) at a lower cost and, when time is priced, in less time.
"""
from __future__ import annotations

from .jev import choice

from .router import BUCKETS, DIFFICULTY, UNTRUSTED

SHRINK = 2.0        # pseudo-problems at the slot's overall accuracy mixed into each bucket's record
MAX_OPTIONS = 10    # options shown per turn
RISK = 0.05         # success-rate shortfall against the reference still treated as the same intelligence

INSTRUCTION = (
    "Choose the solver for `message`. Each option is a model at a reasoning effort, with its measured record "
    "on calibration problems you judged as hard as this one (`difficulty_judged`). The goal is the answer "
    "quality of the reference option at the lowest cost: pick the cheapest option whose record gives no "
    "reason to doubt it on this message. `budget.allowance` is what this turn may cost, in the unit of "
    "`budget.unit`; when no trustworthy option fits it, pick the most reliable option that does fit. "
    "`budget.latency_target_s`, when present, is how fast the answer is wanted. `context` describes the "
    "conversation so far.")
CHECKED = (" A wrong answer will be caught by a check and retried on the reference option, so a cheap option "
           "with a good but imperfect record is worth trying first when it saves enough.")
UNCHECKED = (" A wrong answer will not be caught, so do not gamble on an option that failed problems of this "
             "difficulty.")


def cards(grid, items: list[dict], buckets: dict[str, int]) -> dict:
    """slot -> its record overall and per difficulty bucket, on `items` only."""
    groups = {bucket: [item["id"] for item in items if buckets[item["id"]] == bucket] for bucket in BUCKETS}
    ids = [item["id"] for item in items]
    out = {}
    for slot in grid.slot_ids:
        solved = sum(grid.correct(i, slot) for i in ids)
        overall = solved / len(ids) if ids else 0.0
        mean_cost = sum(grid.cost(i, slot) for i in ids) / len(ids) if ids else 0.0
        mean_wall = sum(grid.wall(i, slot) for i in ids) / len(ids) if ids else 0.0
        per = {}
        for bucket, group in groups.items():
            right, n = sum(grid.correct(i, slot) for i in group), len(group)
            per[bucket] = {
                "n": n, "solved": right,
                # A weak slot can pass two problems by luck: shrink toward what it does overall.
                "p": (right + SHRINK * overall) / (n + SHRINK),
                "cost": sum(grid.cost(i, slot) for i in group) / n if n else mean_cost,
                "wall": sum(grid.wall(i, slot) for i in group) / n if n else mean_wall}
        out[slot] = {"solved": solved, "n": len(ids), "overall": overall, "buckets": per}
    return out


def options(card: dict, bucket: int, reference: str, time_value: float = 0.0, keep=()) -> list[dict]:
    """The slots worth offering for one bucket, cheapest first: each is more reliable than every
    cheaper one. `time_value` prices a second of waiting, so a fast slot can beat a cheap one.
    The reference slot is always offered, and so is every slot named in `keep`."""
    rows = [{"slot": slot, "overall_solved": entry["solved"], "overall_n": entry["n"], **entry["buckets"][bucket]}
            for slot, entry in card.items()]
    for row in rows:
        row["price"] = row["cost"] + time_value * row["wall"]
    rows.sort(key=lambda row: (row["price"], -row["p"]))
    kept, best, forced = [], -1.0, {reference, *keep}
    for row in rows:
        if row["p"] > best + 1e-9 or row["slot"] in forced:
            kept.append(row)
            best = max(best, row["p"])
    while len(kept) > MAX_OPTIONS:
        # Drop the option that adds the least reliability over the one before it.
        gains = [(kept[i]["p"] - kept[i - 1]["p"], i) for i in range(1, len(kept) - 1)
                 if kept[i]["slot"] not in forced]
        if not gains:
            break
        kept.pop(min(gains)[1])
    return kept


def derive(opts: list[dict], reference: str, *, allowance: float | None = None, checked: bool = False,
           risk: float = RISK) -> str:
    """The arithmetic selection from the same cards Jev sees.

    Unchecked: the cheapest option within `risk` of the reference's success rate.
    Checked: the option with the lowest expected price when a failure is retried on the reference.
    An allowance that the choice does not fit swaps it for the most reliable option that does."""
    ref = next(option for option in opts if option["slot"] == reference)
    if checked:
        pick = min(opts, key=lambda o: o["price"] + (0.0 if o is ref else (1.0 - o["p"]) * ref["price"]))
    else:
        pick = min((o for o in opts if o["p"] >= ref["p"] - risk), key=lambda o: o["price"])
    if allowance is not None and pick["cost"] > allowance:
        fits = [o for o in opts if o["cost"] <= allowance]
        pick = max(fits, key=lambda o: (o["p"], -o["price"])) if fits else min(opts, key=lambda o: o["cost"])
    return pick["slot"]


def describe(option: dict, ref: dict, unit: str) -> str:
    ratio = option["cost"] / ref["cost"] if ref["cost"] else 1.0
    tag = " (the reference)" if option is ref else ""
    def record(solved: int, n: int) -> str:
        if not n:
            return "not measured on any"
        return f"right on all {n}" if solved == n else f"failed {n - solved} of {n}"

    return (f"{option['slot']}{tag}: {record(option['solved'], option['n'])} measured problems of this "
            f"difficulty, {record(option['overall_solved'], option['overall_n'])} overall; typical cost "
            f"{option['cost']:.4g} {unit} for a turn like this ({ratio:.2f}x the reference); typical "
            f"{option['wall']:.0f} s")


def turn_state(item: dict, bucket: int, budget: dict, context: dict) -> dict:
    return {"conversation": item["context"], "message": item["message"],
            "difficulty_judged": DIFFICULTY[bucket], "context": context, "budget": budget}


def turn_context(item: dict, **extra) -> dict:
    """What Jev is told about the conversation besides its text."""
    return {"turn": len(item["context"]) + 1, "prior_turns": len(item["context"]),
            "context_tokens_estimate": len(item["prompt"]) // 4, **extra}


def ask(jev, item: dict, bucket: int, opts: list[dict], reference: str, *, budget: dict, context: dict,
        checked: bool) -> str:
    """Jev picks the slot, seeing the cards, the budget and the conversation."""
    if len(opts) < 2:
        return opts[0]["slot"]
    ref = next(option for option in opts if option["slot"] == reference)
    unit = str(budget.get("unit", "usd"))
    criteria = {f"option{i}": describe(option, ref, unit) for i, option in enumerate(opts)}
    question = choice(INSTRUCTION + (CHECKED if checked else UNCHECKED) + UNTRUSTED, criteria)
    answer = jev.ask(turn_state(item, bucket, budget, context), {"select": question})
    return opts[int(answer["select"]["choice"][len("option"):])]["slot"]


def static_budget(ref: dict, unit: str, fraction: float | None, latency_target: float | None = None) -> dict:
    """A per-turn allowance as a fraction of what the reference would cost on a turn like this."""
    budget = {"unit": unit, "reference_cost": round(ref["cost"], 6),
              "allowance": "no cap" if fraction is None else round(fraction * ref["cost"], 6)}
    if latency_target is not None:
        budget["latency_target_s"] = latency_target
    return budget
