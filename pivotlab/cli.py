"""pivot: test Jev as the switch between (model, reasoning-effort) slots.

  pivot slots                 list every slot the two CLIs advertise (--models all: every model)
  pivot smoke                 one trivial problem on every slot
  pivot grid   --name N       run every slot on every problem once (cached, budget-capped)
  pivot replay --name N       score routing policies against the grid (Jev calls only)
  pivot loop   --name N       run a fresh conversation live with Jev pivoting between slots
  pivot report --name N       reprint the tables from saved evidence
  pivot run | fit | doctor | interop
                              the sub-executor other agents spawn: see `pivot run --help`
  pivot scout  --name N --against R --new-since 30
                              screen newly released gateway models against run R's measured slots,
                              stopping each one as soon as a measured slot beats it (billed, capped)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

from . import exec_cli, gateways, scout as scout_mod, slots as slots_mod
from .grid import Breaker, Budget, Cache, run_grid
from .loop import fit, live_thread
from .router import CachedJev, Grid, replay
from .tasks import build_suite

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def patterns(options) -> list[str]:
    return [part for part in (options.slots or "").split(",") if part]


def contenders(run: str, metric: str) -> list[str]:
    """The slots an earlier run's replay shows any policy could pick: its ladder, every option
    offered for some turn, and the reference slots."""
    path = RUNS / run / f"replay-{metric}.json"
    if not path.exists():
        raise ValueError(f"{path} is missing; run `pivot replay --name {run} --metric {metric}` first")
    result = json.loads(path.read_text(encoding="utf-8"))
    found = list(result.get("full_ladder", []))
    for row in result.get("selection", []):
        found += [row["reference"]] + [option["slot"] for option in row["options"]]
    return list(dict.fromkeys(found))


def pick_slots(options) -> list[slots_mod.Slot]:
    efforts = options.efforts.split(",") if options.efforts else None
    found = slots_mod.discover(tuple(options.models.split(",")), efforts)
    keep = patterns(options)
    if options.slots_from:
        wanted = set(contenders(options.slots_from, options.metric))
        found = [slot for slot in found if slot.id in wanted]
    found = [slot for slot in found if not keep or any(fnmatch.fnmatchcase(slot.id, p) for p in keep)]
    if not found:
        raise ValueError("no slots left after --slots")
    return found


def make_budget(options) -> Budget:
    return Budget({"claude": options.max_usd_claude, "codex": options.max_usd_codex,
                   "openrouter": options.max_usd_openrouter, "ramp": options.max_usd_ramp}, options.max_calls)


def slot_table(items: list[dict], rows: list[dict]) -> str:
    tiers = sorted({item["tier"] for item in items})
    tier_of = {item["id"]: item["tier"] for item in items}
    lines = [f"{'slot':<26}{'right':>7}  " + " ".join(f"t{t:<4}" for t in tiers)
             + f"{'out tok':>9}{'wall s':>8}{'usd eq':>9}{'miss':>6}"]
    for slot in dict.fromkeys(row["slot"] for row in rows):
        mine = [row for row in rows if row["slot"] == slot]
        ran = [row for row in mine if row["status"] in ("ok", "timeout")]
        by_tier = []
        for tier in tiers:
            cell = [row for row in ran if tier_of[row["item"]] == tier]
            by_tier.append(f"{sum(r['correct'] for r in cell)}/{len(cell)}".ljust(5))
        okay = [row for row in ran if row["status"] == "ok"]
        mean = lambda values: sum(values) / len(values) if values else 0.0  # noqa: E731
        lines.append(f"{slot:<26}{sum(r['correct'] for r in ran):>3}/{len(ran):<3}  " + " ".join(by_tier)
                     + f"{mean([r['usage']['output'] for r in okay]):>9.0f}"
                     + f"{mean([r['wall_s'] for r in ran]):>8.1f}"
                     + f"{sum(r.get('usd') or 0.0 for r in ran):>9.3f}{len(mine) - len(ran):>6}")
    return "\n".join(lines)


EFFORT_ORDER = ["default", "low", "medium", "high", "xhigh", "max", "ultra"]


def matrix_tables(items: list[dict], rows: list[dict], metric: str, prices: dict) -> str:
    """Model by effort: problems right, then the cost of running the whole suite on that slot."""
    ids = {item["id"] for item in items}
    cells: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["item"] in ids and row["status"] in ("ok", "timeout"):
            cells.setdefault(tuple(row["slot"].rsplit("@", 1)), []).append(row)
    models = list(dict.fromkeys(model for model, _ in cells))
    efforts = [e for e in EFFORT_ORDER if any(effort == e for _, effort in cells)]
    width = max((len(model) for model in models), default=5) + 2

    def money(rows_: list[dict]) -> str:
        try:
            return f"{sum(slots_mod.cost(r, metric, prices) for r in rows_ if r['status'] == 'ok'):.3f}"
        except ValueError:
            return "n/a"

    def table(title: str, cell) -> list[str]:
        out = [title, f"{'model':<{width}}" + "".join(f"{e:>9}" for e in efforts)]
        for model in models:
            out.append(f"{model:<{width}}" + "".join(
                f"{cell(cells[(model, e)]) if (model, e) in cells else '-':>9}" for e in efforts))
        return out

    right = table(f"problems right (of {len(items)})",
                  lambda rs: f"{sum(r['correct'] for r in rs)}/{len(rs)}")
    spent = table(f"cost of the suite, {metric}", money)
    return "\n".join(right + [""] + spent)


def replay_table(result: dict, everything: bool = False) -> str:
    policies = result["policies"]
    unit = {"usd": "usd eq", "usd_flat": "usd flat", "out_tokens": "out tok", "wall_s": "wall s"}[result["metric"]]
    strongest = max((name for name in policies if name.startswith("always:")),
                    key=lambda name: (policies[name]["correct"], -policies[name]["cost"]))
    reference = policies[strongest]["cost"] or 1.0
    order = sorted(policies, key=lambda name: (name.startswith("always:"), -policies[name]["correct"],
                                               policies[name]["cost"]))
    # With a hundred slots, only the fixed slots that matter are listed: the cost/accuracy
    # frontier and the cheapest few at full marks. --all-fixed lists every one.
    hidden: dict[str, int] = {}
    if not everything:
        ladder = {f"always:{slot}" for slot in result.get("full_ladder", [])}
        for family in ("always:", "opus-rule:"):
            names = [name for name in order if name.startswith(family)]
            keep = set(names[:4]) | (ladder & set(names)) | {strongest}
            hidden[family] = len(names) - len(keep & set(names))
            order = [name for name in order if not name.startswith(family) or name in keep]
    clock = policies[strongest]["wall_s"] or 1.0
    lines = [f"{'policy':<44}{'right':>8}{unit:>10}{'cost x':>8}{'calls':>7}{'wall s':>9}{'time x':>8}"]
    for name in order:
        p = policies[name]
        flag = "  <- best fixed slot" if name == strongest else ""
        lines.append(f"{name:<44}{p['correct']:>4}/{p['n']:<3}{p['cost']:>10.3f}"
                     f"{p['cost'] / reference:>7.2f}x{p['attempts']:>7}{p['wall_s']:>9.1f}"
                     f"{p['wall_s'] / clock:>7.2f}x{flag}")
    for family, count in hidden.items():
        if count:
            lines.append(f"({count} more {family}<...> rows hidden; --all-fixed shows them)")
    if result.get("full_ladder"):
        lines.append("cost/accuracy ladder: " + " < ".join(result["full_ladder"]))
    return "\n".join(lines)


def diagnostics_table(result: dict) -> str:
    lines = [f"{'item':<24}{'tier':>5}{'jev diff':>10}{'lookup':>8}{'slots wrong':>13}  cheapest correct"]
    for d in result["diagnostics"]:
        lines.append(f"{d['item']:<24}{d['tier']:>5}{d['jev_difficulty']:>10.2f}{d['jev_lookup']:>8.2f}"
                     f"{d['slots_wrong']:>13}  {d['cheapest_correct'] or 'none'}")
    rho = result["rank_correlation"]
    fmt = lambda value: "n/a" if value is None else f"{value:+.2f}"  # noqa: E731
    lines.append(f"rank correlation with slots-wrong: jev difficulty {fmt(rho['jev_difficulty_vs_slots_wrong'])}, "
                 f"generator tier {fmt(rho['tier_label_vs_slots_wrong'])}")
    return "\n".join(lines)


def cmd_slots(options) -> int:
    for slot in pick_slots(options):
        print(f"{slot.provider:<11}{slot.id}")
    return 0


def cmd_grid(options) -> int:
    out = RUNS / options.name
    out.mkdir(parents=True, exist_ok=True)
    per_tier = (1, 0, 0, 0, 0) if options.command == "smoke" else tuple(int(x) for x in options.per_tier.split(","))
    extras = [tuple(part.split(":")) for part in options.extra.split(",")] if options.extra else []
    if any(len(extra) != 3 for extra in extras):
        raise ValueError("--extra takes seed:family:count entries separated by commas")
    if options.items_from and options.command != "smoke":
        # The same problems as an earlier run, so its paid cells are reused from the cache.
        items = json.loads((RUNS / options.items_from / "items.json").read_text(encoding="utf-8"))
    else:
        items = build_suite(options.seed, per_tier, 0 if options.command == "smoke" else options.threads, extras)
    if options.take:
        items = items[:options.take]
    chosen, budget, breaker = pick_slots(options), make_budget(options), Breaker()
    log(f"[grid] {len(items)} items x {len(chosen)} slots = {len(items) * len(chosen)} cells; "
        f"caps ${options.max_usd_claude} claude / ${options.max_usd_codex} codex (API-equivalent), "
        f"{options.max_calls} calls, {options.timeout}s per call")
    metered = sorted({slot.provider for slot in chosen if slot.metered})
    if metered:
        log(f"[grid] BILLED providers in this run: {metered}; hard caps ${options.max_usd_openrouter} openrouter / "
            f"${options.max_usd_ramp} ramp, each call reserved at its worst case before it is sent")
    rows = run_grid(items, chosen, cache=Cache(RUNS / "cache" / "slot_calls.jsonl"), budget=budget,
                    prices=slots_mod.load_prices(), timeout=options.timeout, concurrency=options.concurrency,
                    breaker=breaker, log=log)
    write_json(out / "items.json", items)
    (out / "grid.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                    encoding="utf-8")
    write_json(out / "grid_budget.json", {**budget.summary(), **breaker.summary()})
    print(slot_table(items, rows) if len(chosen) <= 24 else
          matrix_tables(items, rows, "out_tokens", slots_mod.load_prices()))
    stopped = breaker.summary()
    for slot, reason in stopped["stopped_slots"].items():
        log(f"[grid] not served: {slot}: {reason}")
    if stopped["stopped_providers"]:
        log(f"[grid] STOPPED EARLY on {stopped['stopped_providers']}: errors in a row, likely a rate limit; "
            f"rerun the same command later to resume from the cache")
    left = sum(row["status"] == "skipped" for row in rows)
    if left:
        log(f"[grid] {left} cells not run")
    used_tools = sorted({row["slot"] for row in rows if row.get("tool_events")})
    if used_tools:
        log(f"[grid] WARNING: tool activity seen on {used_tools}; those answers are not reasoning-only")
    log(f"[grid] fresh spend this run: {json.dumps(budget.summary())}")
    log(f"evidence: {out}")
    return 0


def load_run(options) -> tuple[Path, list[dict], list[dict]]:
    out = RUNS / options.name
    if not (out / "grid.jsonl").exists():
        raise SystemExit(f"{out} has no grid.jsonl; run `pivot grid --name {options.name}` first")
    return out, json.loads((out / "items.json").read_text(encoding="utf-8")), read_jsonl(out / "grid.jsonl")


def make_jev(options) -> CachedJev:
    from .jev import JevClient
    return CachedJev(JevClient(model=options.jev_model, max_attempts=options.max_jev_requests),
                     RUNS / "cache" / "jev_calls.jsonl")


def cmd_replay(options) -> int:
    out, items, rows = load_run(options)
    grid = Grid(items, rows, options.metric, slots_mod.load_prices(), patterns(options))
    if len(grid.items) < 3:
        raise SystemExit(f"only {len(grid.items)} complete items; the grid needs at least 3")
    log(f"[replay] {len(grid.slot_ids)} slots x {len(grid.items)} items, metric {options.metric}")
    calibration = None
    if options.fit_from:
        fit_dir = RUNS / options.fit_from
        calibration = Grid(json.loads((fit_dir / "items.json").read_text(encoding="utf-8")),
                           read_jsonl(fit_dir / "grid.jsonl"), options.metric, slots_mod.load_prices(),
                           grid.slot_ids)
        if set(calibration.slot_ids) != set(grid.slot_ids):
            raise SystemExit(f"{options.fit_from} lacks slots this run has: "
                             f"{sorted(set(grid.slot_ids) - set(calibration.slot_ids))}")
        log(f"[replay] held-out test: every fit comes from {options.fit_from} "
            f"({len(calibration.items)} problems), none from this run")
    if grid.dead_slots:
        log(f"[replay] left out, never answered: {grid.dead_slots}")
    if grid.unpriced_slots:
        log(f"[replay] left out, no list price: {grid.unpriced_slots}")
    result = replay(grid, make_jev(options), log, options.time_value, calibration)
    tag = f"-{options.tag}" if options.tag else ""
    write_json(out / f"replay-{options.metric}{tag}.json", result)
    print(replay_table(result, options.all_fixed))
    print()
    print(diagnostics_table(result))
    j = result["jev"]
    log(f"[replay] jev {j['resolved_models']}: {j['fresh']['requests']} fresh requests, "
        f"{j['fresh']['input_tokens']} input tokens, about ${j['estimated_fresh_usd']:.4f}")
    if grid.dropped:
        log(f"[replay] dropped incomplete items: {grid.dropped}")
    if grid.imputed:
        log(f"[replay] {grid.imputed} timed-out cells charged at the slot's worst observed cost")
    log(f"evidence: {out / f'replay-{options.metric}{tag}.json'}")
    return 0


def cmd_loop(options) -> int:
    out, items, rows = load_run(options)
    grid = Grid(items, rows, options.metric, slots_mod.load_prices(), patterns(options))
    jev = make_jev(options)
    fitted = fit(grid, jev, options.time_value)
    by_id = {row["slot"]: slots_mod.Slot(
        row["provider"], row["model"], row["effort"], 0,
        tuple(row["price"][key] for key in ("input", "cached_input", "output")) if row.get("price") else None)
        for row in rows if row.get("provider")}
    if options.route in ("select", "derived"):
        log(f"[loop] reference (the intelligence to match): {fitted['reference']}")
        for bucket, opts in fitted["options"].items():
            log(f"[loop] difficulty {bucket} options: " + ", ".join(
                f"{o['slot']} {o['solved']}/{o['n']} {o['cost']:.4f}" for o in opts))
    elif options.route == "ladder":
        log(f"[loop] ladder: {' < '.join(fitted['ladder'])}")
        log(f"[loop] difficulty bucket -> rung: { {b: fitted['ladder'][r] for b, r in fitted['map_jev'].items()} }")
    elif options.route == "cheapest":
        log(f"[loop] ladder: {' < '.join(fitted['ladder'])}")
        log(f"[loop] difficulty bucket -> slot: {fitted['cheap']}")
    else:
        log(f"[loop] difficulty bucket -> slot: {fitted['best']}")
        log(f"[loop] escalation order: {fitted['reliable']}")
    common = {"cache": Cache(RUNS / "cache" / "slot_calls.jsonl"), "budget": make_budget(options),
              "prices": slots_mod.load_prices(), "timeout": options.timeout, "metric": options.metric,
              "kind": options.thread_kind, "log": log}
    runs = [live_thread(options.thread_seed, 0, fitted, by_id, jev, route=options.route,
                        escalate=not options.no_escalate, hysteresis=not options.no_hysteresis,
                        budget_x=options.budget_x, latency_target=options.latency_target, **common)]
    if options.baseline:
        slot = fitted["reliable"][0] if options.baseline == "top" else options.baseline
        log(f"[loop] baseline: every turn on {slot}")
        runs.append(live_thread(options.thread_seed, 0, fitted, by_id, jev, fixed=slot, **common))
    cap = "" if options.budget_x is None else f"-{options.budget_x}x"
    stem = f"loop-{options.route}{cap}-{options.thread_kind}-{options.thread_seed}-{options.metric}.json"
    write_json(out / stem, {"fitted": fitted, "runs": runs})
    for run in runs:
        t = run["totals"]
        print(f"{run['mode']:<28} right {t['correct']}/{t['n']}  {t['metric']} {t['cost']:.3f}  "
              f"observed usd eq {t['usd']:.3f}  out tok {t['out_tokens']}  calls {t['attempts']}  "
              f"wall {t['wall_s']}s")
    log(f"evidence: {out / stem}")
    return 0


def cmd_scout(options) -> int:
    if not options.against:
        raise ValueError("scout needs --against RUN: the run whose measured slots a new model must beat")
    ref = RUNS / options.against
    items, rows = json.loads((ref / "items.json").read_text(encoding="utf-8")), read_jsonl(ref / "grid.jsonl")
    prices, cache = slots_mod.load_prices(), Cache(RUNS / "cache" / "slot_calls.jsonl")
    grid = Grid(items, rows, options.metric, prices)
    if options.new_since is not None:
        tested = {(receipt.get("provider"), receipt.get("model")) for receipt in cache.rows.values()}
        skip = tuple(part for part in options.skip_vendors.split(",") if part)
        specs = []
        for provider, spec in gateways.GATEWAYS.items():
            if not os.environ.get(spec["key"]):
                log(f"[scout] {provider}: {spec['key']} is not set, not searched")
                continue
            found = scout_mod.new_models(provider, gateways.catalog(provider, slots_mod.SCRATCH),
                                         options.new_since, skip, tested)
            log(f"[scout] {provider}: {len(found)} untested models listed in the last {options.new_since:g} days")
            specs += [f"{provider}:{model}" for model in found]
        if not specs:
            print("no untested models in that window")
            return 0
        options.models = ",".join(specs)
    elif options.models == ",".join(slots_mod.DEFAULT_MODELS):
        raise ValueError("scout needs --new-since DAYS or --models with the models to screen")
    chosen = [slot for slot in pick_slots(options) if slot.id not in grid.slot_ids]
    budget, breaker = make_budget(options), Breaker()
    log(f"[scout] {len(chosen)} new slots over {len({(s.provider, s.model) for s in chosen})} models against "
        f"{len(grid.slot_ids)} measured slots on {len(grid.items)} problems; BILLED hard caps "
        f"${options.max_usd_openrouter} openrouter / ${options.max_usd_ramp} ramp")
    results = scout_mod.scout(chosen, grid, metric=options.metric, prices=prices, cache=cache, budget=budget,
                              breaker=breaker, timeout=options.timeout, concurrency=options.concurrency, log=log)
    out = RUNS / options.name
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "scout.json", {
        "against": options.against, "metric": options.metric, "problems": len(grid.items),
        "results": [{key: value for key, value in row.items() if key != "cells"} for row in results],
        "spend": budget.summary()})
    # The incumbents plus every new slot that ran every problem: a grid any other command can replay.
    kept = {item["id"] for item in grid.items}
    finished = [row for row in results if row["ran"] == len(grid.items)]
    merged = [row for row in rows if row["item"] in kept and row["slot"] in grid.slot_ids]
    merged += [cell for row in finished for cell in row["cells"]]
    write_json(out / "items.json", grid.items)
    (out / "grid.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
                                    encoding="utf-8")
    unit = {"usd": "usd eq", "usd_flat": "usd flat", "out_tokens": "out tok", "wall_s": "wall s"}[options.metric]
    print(scout_mod.table(results, unit))
    log(f"[scout] fresh spend this run: {json.dumps(budget.summary())}")
    log(f"[scout] {len(finished)} new slots ran every problem and are in {out / 'grid.jsonl'} for replay")
    log(f"evidence: {out / 'scout.json'}")
    return 0


def cmd_report(options) -> int:
    out, items, rows = load_run(options)
    keep = patterns(options)
    rows = [row for row in rows if not keep or any(fnmatch.fnmatchcase(row["slot"], p) for p in keep)]
    if len({row["slot"] for row in rows}) <= 24:
        print(slot_table(items, rows))
    else:
        print(matrix_tables(items, rows, options.metric, slots_mod.load_prices()))
    for path in sorted(out.glob("replay-*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n== {path.name} ==\n{replay_table(result, options.all_fixed)}\n\n{diagnostics_table(result)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in exec_cli.COMMANDS:       # the sub-executor has its own, stricter command line
        return exec_cli.main(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["slots", "smoke", "grid", "replay", "loop", "report", "scout"])
    parser.add_argument("--name", default="pilot", help="run directory under runs/")
    parser.add_argument("--models", default=",".join(slots_mod.DEFAULT_MODELS),
                        help="claude:<model>,codex:<model>,...; `all`, `claude:all`, `codex:all` take every "
                             "model in catalog.json / the Codex model cache; `current` the newest of each line; "
                             "openrouter:<vendor/model> and ramp:<model> are metered gateway slots")
    parser.add_argument("--slots", help="comma list of slot-id globs to keep, e.g. 'claude-*@high,gpt-6-*'")
    parser.add_argument("--items-from", help="reuse the problems of an earlier run instead of building a suite")
    parser.add_argument("--slots-from", help="grid: only the slots an earlier run's replay shows a policy "
                                             "could pick (its ladder, offered options and reference)")
    parser.add_argument("--fit-from", help="replay: fit everything on this other run's grid, making the "
                                           "replayed run a held-out test")
    parser.add_argument("--take", type=int, help="only the first N problems of the suite")
    parser.add_argument("--tag", help="suffix for the replay evidence file, to keep several slot subsets apart")
    parser.add_argument("--against", help="scout: the run whose measured slots a new model is judged against")
    parser.add_argument("--new-since", type=float,
                        help="scout: screen every gateway model listed in the last N days that was never called")
    parser.add_argument("--skip-vendors", default="anthropic,openai",
                        help="scout --new-since: vendors already measured through the subscription CLIs")
    parser.add_argument("--all-fixed", action="store_true", help="list every fixed slot in replay tables")
    parser.add_argument("--efforts", help="comma list to keep; default is every effort each CLI advertises")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--per-tier", default="1,1,2,3,2", help="problems per difficulty tier 0..4")
    parser.add_argument("--threads", type=int, default=1, help="six-turn conversations to add to the grid")
    parser.add_argument("--extra", help="more problems as seed:family:count,... (see tasks.FAMILIES)")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-usd-claude", type=float, default=6.0, help="API-equivalent cap, fresh calls")
    parser.add_argument("--max-usd-codex", type=float, default=6.0, help="API-equivalent cap, fresh calls")
    parser.add_argument("--max-usd-openrouter", type=float, default=0.25,
                        help="BILLED dollars: hard cap on OpenRouter spend in this run")
    parser.add_argument("--max-usd-ramp", type=float, default=0.25,
                        help="BILLED dollars: hard cap on Ramp Router spend in this run")
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--metric", default="usd_flat", choices=["usd", "usd_flat", "out_tokens", "wall_s"],
                        help="usd_flat: list price with no cache discount; usd: API-equivalent as observed")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--max-jev-requests", type=int, default=1500)
    parser.add_argument("--time-value", type=float, default=0.0,
                        help="what a second of waiting is worth, in the cost unit; above 0 the selection "
                             "trades money for speed")
    parser.add_argument("--budget-x", type=float,
                        help="loop: the conversation's budget as a multiple of what the reference slot would "
                             "cost; Jev is told what is left before every turn")
    parser.add_argument("--latency-target", type=float, help="loop: seconds per answer Jev is told to aim for")
    parser.add_argument("--thread-seed", default="live-1", help="seed of the fresh conversation for `loop`")
    parser.add_argument("--thread-kind", default="list", choices=["list", "heavy"],
                        help="list: the grid's conversation shape; heavy: two long computations per conversation")
    parser.add_argument("--baseline", help="a slot id, or `top` for the cheapest slot that was right on "
                                           "every grid problem: also run the conversation on that one slot")
    parser.add_argument("--route", default="select", choices=["select", "derived", "cheapest", "bucket", "ladder"],
                        help="select: Jev picks from measured slot cards knowing the budget and the "
                             "conversation; derived: the same cards and budget by arithmetic; cheapest: "
                             "cheapest slot that solved the Jev difficulty bucket, climbing the ladder on a "
                             "failed check; bucket: cheapest slot right on every grid problem; ladder: jev-opus")
    parser.add_argument("--no-escalate", action="store_true")
    parser.add_argument("--no-hysteresis", action="store_true")
    options = parser.parse_args(argv)
    if options.command == "smoke" and options.name == "pilot":
        options.name = "smoke"
    handler = {"slots": cmd_slots, "smoke": cmd_grid, "grid": cmd_grid, "replay": cmd_replay,
               "loop": cmd_loop, "report": cmd_report, "scout": cmd_scout}[options.command]
    try:
        return handler(options)
    except (RuntimeError, ValueError) as error:      # JevError is a RuntimeError
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
