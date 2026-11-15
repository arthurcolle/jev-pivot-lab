"""Intake for models nobody has measured yet: is a new slot good, expensive or crap?

A new slot runs the problems of a reference grid and is judged at checkpoints against the
slots already measured on that whole grid. It stops as soon as one of them beats it outright,
so a weak or overpriced model costs a handful of calls rather than a grid.

Only an incumbent that the new slot can no longer out-score counts as a rival: one whose wrong
answers over the whole grid number no more than the new slot's so far. Cost is compared on the
problems the new slot has actually run.

  good       ran every problem, as accurate as the best measured slot, and no rival was cheaper
  tradeoff   ran every problem and nothing as accurate is cheaper, but it is less accurate than
             the best measured slot: a new cheap point, not a replacement
  beaten     a rival is at least as accurate for less, or more accurate for no more
  expensive  a rival does the same problems for a fraction of the cost (a tenth on the warm-up,
             a third after eight problems, two thirds after sixteen)
  crap       beaten on accuracy, and wrong on more than half of what it ran (hard problems come
             first, so this is a slot that cannot do the work, not one that is merely second best)
  broken     the gateway keeps erroring on it

The first three problems are cheap ones every incumbent solves: they catch a broken model and
a hopeless price (ten times a rival) for pennies. The rest come most-discriminating first, so
weakness and the real cost of hard problems both show early.

Efforts of one model are bracketed instead of all run: `high` first, then the lowest effort,
and the levels between or above only when what was measured leaves them a chance.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from . import gateways
from .grid import run_cell
from .slots import Slot, cost as receipt_cost

CHECKPOINTS = (3, 8, 16)
WARMUP = 3
# How many times a rival's cost a new slot must reach to be stopped as expensive, by problems run.
# On the easy warm-up a dear model's token thrift cannot show yet, so only a hopeless price stops it
# (the thriftiest strong model measured used about a seventh of the tokens of the cheapest reliable
# one). The bar tightens as the hard problems, where most of the cost is, come in.
MARGINS = ((3, 10.0), (8, 3.0), (10 ** 9, 1.5))
CRAP_BELOW = 0.5     # share right, under which a slot beaten on accuracy is called crap rather than beaten


def order_items(grid) -> list[dict]:
    wrong = {item["id"]: sum(not grid.correct(item["id"], slot) for slot in grid.slot_ids) for item in grid.items}
    easy = sorted((item for item in grid.items if wrong[item["id"]] == 0),
                  key=lambda item: sum(grid.cost(item["id"], slot) for slot in grid.slot_ids))[:WARMUP]
    rest = sorted((item for item in grid.items if item not in easy),
                  key=lambda item: (-wrong[item["id"]], -item["tier"]))
    return easy + rest


def standing(grid, cells: list[dict], metric: str, prices: dict, final: bool) -> dict:
    """Where a new slot stands against the measured slots it can no longer out-score."""
    ids = [cell["item"] for cell in cells]
    right = sum(bool(cell["correct"]) for cell in cells)
    wrong, total = len(cells) - right, len(grid.items)
    spent = sum(receipt_cost(cell, metric, prices) for cell in cells if cell["status"] == "ok")
    margin = next(factor for limit, factor in MARGINS if len(cells) <= limit)
    rivals = []
    for slot in grid.slot_ids:
        missed = sum(not grid.correct(item["id"], slot) for item in grid.items)
        if missed <= wrong:
            rivals.append((slot, missed, sum(grid.cost(i, slot) for i in ids)))
    cheapest = min(rivals, key=lambda rival: rival[2]) if rivals else None
    stronger = [rival for rival in rivals if rival[1] <= wrong - (1 if final else 2) and rival[2] <= spent]
    best = min(sum(not grid.correct(item["id"], slot) for item in grid.items) for slot in grid.slot_ids)
    settled = True
    if stronger:
        verdict = "crap" if right < CRAP_BELOW * len(cells) else "beaten"
        rival = min(stronger, key=lambda r: (r[1], r[2]))
    elif spent > 0 and cheapest and cheapest[2] * margin <= spent:
        verdict, rival = "expensive", cheapest
    elif final and cheapest and cheapest[2] < spent:
        verdict, rival = "beaten", cheapest
    elif final:
        verdict, rival = ("good" if wrong <= best else "tradeoff"), cheapest
    else:
        verdict, rival, settled = "alive", cheapest, False
    return {"verdict": verdict, "settled": settled, "ran": len(cells), "right": right, "cost": spent,
            "wall_s": round(sum(cell.get("wall_s", 0.0) for cell in cells), 1),
            "rival": rival[0] if rival else None, "rival_right": total - rival[1] if rival else None,
            "rival_cost": rival[2] if rival else None, "of": total}


def screen(slot: Slot, grid, order: list[dict], *, metric: str, prices: dict, run) -> dict:
    """Run one new slot through the problems until a checkpoint settles it. `run(item, slot)` -> cell."""
    cells: list[dict] = []
    for n, item in enumerate(order, 1):
        cell = run(item, slot)
        if cell["status"] == "skipped":
            state = standing(grid, cells, metric, prices, False) if cells else {"ran": 0, "right": 0, "cost": 0.0}
            broken = str(cell.get("reason", "")).startswith("slot stopped")
            return {**state, "verdict": "broken" if broken else "unfinished", "note": cell.get("reason"),
                    "cells": cells}
        cells.append(cell)
        if n <= WARMUP and sum(c["status"] == "error" for c in cells) >= 2:
            return {**standing(grid, cells, metric, prices, False), "verdict": "broken",
                    "note": str(cell.get("error"))[:160], "cells": cells}
        if n in CHECKPOINTS and n < len(order):
            state = standing(grid, cells, metric, prices, False)
            if state["settled"]:
                return {**state, "cells": cells}
    return {**standing(grid, cells, metric, prices, True), "cells": cells}


def bracket(levels: list[Slot]) -> list[Slot]:
    """The order a model's efforts are tried in: `high` (or its top level), the lowest, then the rest."""
    anchor = next((slot for slot in levels if slot.effort == "high"), levels[-1])
    return list(dict.fromkeys([anchor, levels[0]] + levels))


def skip_reason(slot: Slot, levels: list[Slot], done: dict) -> str | None:
    """Why an effort is not worth its calls given what the model's other efforts showed."""
    anchor, lowest = bracket(levels)[0], levels[0]
    top, low = done.get(anchor.id), done.get(lowest.id)
    if slot is anchor or top is None:
        return None
    if top["verdict"] in ("crap", "broken"):
        return f"{anchor.effort} effort is already {top['verdict']}"
    if slot.rank > anchor.rank:
        if top["verdict"] in ("expensive", "beaten"):
            return f"costs more than {anchor.effort} effort, which is already {top['verdict']}"
        return f"{anchor.effort} effort already answers everything" if top["right"] == top["ran"] else None
    if slot is lowest or low is None:
        return None
    if low["verdict"] == "expensive":
        return f"{lowest.effort} effort is already expensive"
    if low["verdict"] == "good" and low["right"] == low["ran"]:
        return f"{lowest.effort} effort already answers everything"
    return None


def scout_model(levels: list[Slot], grid, order: list[dict], *, metric: str, prices: dict, run, log) -> list[dict]:
    done: dict[str, dict] = {}
    out = []
    for slot in bracket(levels):
        reason = skip_reason(slot, levels, done)
        if reason:
            result = {"verdict": "not run", "note": reason, "ran": 0, "right": 0, "cost": 0.0, "cells": []}
        else:
            result = done[slot.id] = screen(slot, grid, order, metric=metric, prices=prices, run=run)
        out.append({"slot": slot.id, "provider": slot.provider, "model": slot.model, "effort": slot.effort,
                    "rank": slot.rank, **result})
        ratio = (f"  {result['cost'] / result['rival_cost']:.2f}x {result['rival']}"
                 if result.get("rival_cost") else "")
        log(f"[scout] {slot.id:<52} {result['verdict']:<10} {result['right']}/{result['ran']}"
            f"  {result['cost']:.4f}{ratio}{'  ' + str(result['note']) if result.get('note') else ''}")
    return sorted(out, key=lambda row: row["rank"])


def scout(slots: list[Slot], grid, *, metric: str, prices: dict, cache, budget, breaker, timeout: float,
          concurrency: int = 6, runner=None, log=print) -> list[dict]:
    """Every new model is screened on its own thread; its efforts and problems run in order."""
    order = order_items(grid)
    kwargs = {"cache": cache, "budget": budget, "prices": prices, "timeout": timeout, "breaker": breaker}
    if runner is not None:
        kwargs["runner"] = runner

    def run(item, slot):
        return run_cell(item, slot, **kwargs)

    models: dict[tuple[str, str], list[Slot]] = {}
    for slot in slots:
        models.setdefault((slot.provider, slot.model), []).append(slot)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        batches = list(pool.map(lambda levels: scout_model(sorted(levels, key=lambda s: s.rank), grid, order,
                                                           metric=metric, prices=prices, run=run, log=log),
                                models.values()))
    return [row for batch in batches for row in batch]


def new_models(provider: str, listed: dict, days: float, skip_vendors=(), tested=()) -> list[str]:
    """Models a gateway listed in the last `days` days that this lab has never called.
    Pricing variants (`:free`, `:batch`) and router aliases are left out."""
    cutoff = time.time() - days * 86400
    found = [(entry.get("created") or 0, model) for model, entry in listed.items()
             if (entry.get("created") or 0) >= cutoff and ":" not in model and not model.startswith("~")
             and model.split("/")[0] not in tuple(skip_vendors) + ("openrouter",)
             and entry.get("text", True) and (provider, model) not in set(tested)]
    return [model for _, model in sorted(found, reverse=True)]


def table(rows: list[dict], unit: str) -> str:
    lines = [f"{'slot':<54}{'verdict':<11}{'right':>7}{unit:>10}  against the measured slot it has to beat"]
    for row in rows:
        if row["verdict"] == "not run":
            lines.append(f"{row['slot']:<54}{'not run':<11}{'':>17}  {row['note']}")
            continue
        versus = (f"{row['cost'] / row['rival_cost']:.2f}x the cost of {row['rival']} on the same problems "
                  f"({row['rival_right']}/{row['of']} overall)" if row.get("rival_cost") else
                  (f"{row['rival']} ({row['rival_right']}/{row['of']} overall)" if row.get("rival")
                   else "no measured slot is as accurate"))
        note = f"  [{row['note']}]" if row.get("note") else ""
        lines.append(f"{row['slot']:<54}{row['verdict']:<11}{row['right']:>4}/{row['ran']:<2}{row['cost']:>10.4f}"
                     f"  {versus}{note}")
    return "\n".join(lines)


__all__ = ["scout", "screen", "standing", "order_items", "bracket", "skip_reason", "new_models", "table",
           "gateways"]
