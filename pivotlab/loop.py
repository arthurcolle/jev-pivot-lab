"""The live conversational loop: before each turn Jev picks the slot; a failed check escalates.

Replies come from whichever slot answered, so one conversation moves between models and
efforts. The deterministic answer check stands in for a failing test: a real conversation
needs some other verdict to trigger escalation.
"""
from __future__ import annotations

from . import meta
from .grid import run_cell
from .router import (apply_hysteresis, bucket_best, bucket_cheapest, bucket_map, bucket_of, build_ladder, climb,
                     jev_state, profile_questions, slot_stats)
from .slots import cost as receipt_cost
from .tasks import thread_items, thread_turns


def fit(grid, jev, time_value: float = 0.0) -> dict:
    """Ladder, difficulty mapping and slot cards fitted on the whole grid, for turns it has never seen."""
    buckets = {item["id"]: bucket_of(jev.ask(jev_state(item), profile_questions())) for item in grid.items}
    stats = slot_stats(grid, grid.items)
    ladder = build_ladder(stats)
    reliable = bucket_best(grid, stats, grid.items, buckets)
    cheap = bucket_cheapest(grid, grid.items, buckets, reliable["best"])
    costs = {slot: stats[slot]["cost"] for slot in grid.slot_ids}
    reference, card = reliable["reliable"][0], meta.cards(grid, grid.items, buckets)
    return {"reference": reference, "reference_cost": stats[reference]["cost"], "unit": grid.metric,
            "options": {bucket: meta.options(card, bucket, reference, time_value) for bucket in card[reference]["buckets"]},
            "ladder": ladder, "map_jev": bucket_map(grid, ladder, grid.items, buckets),
            "stats": {slot: stats[slot] for slot in ladder}, **reliable, "cheap": cheap,
            "cheap_chain": {bucket: [slot] + climb(ladder, costs, slot) for bucket, slot in cheap.items()}}


def live_budget(fitted: dict, ref: dict, *, budget_x, turns: int, turn: int, spent: float,
                latency_target) -> tuple[dict, float | None]:
    """What Jev is told about money before a turn, and the allowance in the cost unit.

    The conversation's budget is `budget_x` times what the reference slot would cost over its
    turns. What is left is split in proportion to what the reference would need: a heavy turn
    gets a heavy share, and money saved on easy turns carries forward to later ones."""
    state = {"unit": fitted["unit"], "reference_cost": round(ref["cost"], 6), "spent": round(spent, 6),
             "turns_left_after_this": turns - turn - 1, "allowance": "no cap"}
    allowance = None
    if budget_x is not None:
        total = budget_x * fitted["reference_cost"] * turns
        remaining = max(0.0, total - spent)
        later = fitted["reference_cost"] * (turns - turn - 1)
        allowance = remaining * ref["cost"] / (ref["cost"] + later) if ref["cost"] + later else remaining
        state.update(total=round(total, 6), remaining=round(remaining, 6), allowance=round(allowance, 6))
    if latency_target is not None:
        state["latency_target_s"] = latency_target
    return state, allowance


def live_thread(seed, index: int, fitted: dict, slots: dict, jev, *, cache, budget, prices,
                timeout: float = 300.0, metric: str = "usd", route: str = "bucket", escalate: bool = True,
                hysteresis: bool = True, fixed: str | None = None, kind: str = "list", runner=None,
                budget_x: float | None = None, latency_target: float | None = None, log=print) -> dict:
    """route=select: Jev picks the slot from the measured slot cards for the turn's difficulty,
    knowing the running budget and the conversation; a failed check retries on the reference slot.
    route=derived: the same cards and budget, picked by arithmetic instead of Jev.
    route=cheapest: Jev's difficulty bucket picks the cheapest slot of any strength that solved
    that bucket in the grid; a failed check climbs the cheap-to-strong ladder from there.
    route=bucket: the bucket picks the cheapest slot that was right on every grid problem.
    route=ladder: the jev-opus shape, one cheap-to-strong ladder with hold-after-raise hysteresis."""
    ladder = fitted["ladder"]
    answers, turns, current, hold, spent, previous = [], [], None, 0, 0.0, {}
    count = len(thread_turns(seed, index, kind))
    kwargs = {"cache": cache, "budget": budget, "prices": prices, "timeout": timeout}
    if runner is not None:
        kwargs["runner"] = runner
    for k in range(len(thread_turns(seed, index, kind))):
        item = thread_items(seed, index, answers, kind)[k]
        record = {"turn": k + 1, "item": item["id"], "family": item["family"], "tier": item["tier"]}
        chain = [fixed]
        if not fixed:
            profile = jev.ask(jev_state(item), profile_questions())
            bucket = bucket_of(profile)
            if route == "ladder":
                rung = target = fitted["map_jev"][bucket]
                if hysteresis and current is not None:
                    rung, hold = apply_hysteresis(current, target, hold)
                chain, first = ladder[rung:], ladder[target]
            elif route in ("select", "derived"):
                opts, reference = fitted["options"][bucket], fitted["reference"]
                ref = next(option for option in opts if option["slot"] == reference)
                money, allowance = live_budget(fitted, ref, budget_x=budget_x, turns=count, turn=k, spent=spent,
                                               latency_target=latency_target)
                if route == "select":
                    first = meta.ask(jev, item, bucket, opts, reference, budget=money,
                                     context=meta.turn_context(item, **previous), checked=escalate)
                else:
                    first = meta.derive(opts, reference, allowance=allowance, checked=escalate)
                chain = [first] + [reference] * (first != reference)
                record.update(allowance=money["allowance"], offered=[option["slot"] for option in opts])
            elif route == "cheapest":
                chain = list(fitted["cheap_chain"][bucket])
                first = chain[0]
            else:
                first = fitted["best"][bucket]
                chain = [first] + [slot for slot in fitted["reliable"] if slot != first]
            record.update(jev_difficulty=profile["difficulty"]["score"], jev_lookup=profile["lookup"]["noul"],
                          bucket=bucket, target=first)
        attempts = []
        steps = chain if escalate else chain[:1]
        for step, slot in enumerate(steps):
            cell = run_cell(item, slots[slot], **kwargs)
            attempts.append(cell)
            if cell["correct"] or cell["status"] == "skipped":
                break
            hold = 1 if step + 1 < len(steps) else hold      # a raise is held for one more turn
        if not fixed and route == "ladder":
            current = ladder.index(attempts[-1]["slot"])
        final = attempts[-1]
        answers.append(final.get("text") or "ANSWER: unknown")
        record.update(tried=[cell["slot"] for cell in attempts], correct=final["correct"],
                      status=final["status"], usd=sum(cell.get("usd") or 0.0 for cell in attempts),
                      cost=sum(receipt_cost(cell, metric, prices) for cell in attempts if cell["status"] == "ok"),
                      out_tokens=sum((cell.get("usage") or {}).get("output", 0) for cell in attempts),
                      wall_s=round(sum(cell.get("wall_s", 0.0) for cell in attempts), 1),
                      answer=(final.get("text") or "")[:120], gold=item["gold"])
        turns.append(record)
        spent += record["cost"]
        previous = {"previous_slot": final["slot"], "previous_check_passed": bool(final["correct"]),
                    "previous_attempts": len(attempts)}
        log(f"[turn {k + 1}] tier {item['tier']} {item['family']:<20} -> {' -> '.join(record['tried'])}  "
            f"{'pass' if final['correct'] else 'FAIL'}  ${record['usd']:.3f}  {record['wall_s']}s")
    return {"thread_seed": str(seed), "thread_index": index, "thread_kind": kind,
            "mode": fixed or f"jev-pivot ({route})", "budget_x": None if fixed else budget_x,
            "escalate": escalate and not fixed, "hysteresis": hysteresis and not fixed and route == "ladder",
            "turns": turns,
            "totals": {"correct": sum(t["correct"] for t in turns), "n": len(turns),
                       "usd": round(sum(t["usd"] for t in turns), 4), "metric": metric,
                       "cost": round(sum(t["cost"] for t in turns), 4),
                       "out_tokens": sum(t["out_tokens"] for t in turns),
                       "wall_s": round(sum(t["wall_s"] for t in turns), 1),
                       "attempts": sum(len(t["tried"]) for t in turns)}}
