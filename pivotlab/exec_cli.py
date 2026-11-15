"""pivot run | fit | doctor | interop: the sub-executor's command line.

  pivot run [-] [TEXT]        answer one prompt (argument, --prompt-file, or stdin) on the slot the
                              policy and Jev pick; prints the answer, or one JSON object with --json
  pivot fit --from RUN        compile a measured grid run into the policy file `run` reads
  pivot doctor [--probe]      report whether this machine can run requests, as JSON
  pivot interop               print the exact command another agent (dsco) should spawn

`run` writes nothing but its result to stdout and nothing at all to stderr unless --verbose,
because callers such as `dsco interop run` capture both streams as one. Exit codes follow
dsco's headless convention: 0 answered, 1 run error, 2 usage, 78 config, 143 terminated.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from . import executor, slots as slots_mod, state
from .policy import DEFAULT_PATH, PolicyError, compile_policy, load_policy

COMMANDS = ("run", "fit", "doctor", "interop")
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
JOURNAL = state.STATE / "executions.jsonl"


class FakeJev:
    """Test seam: Jev scripted by the JSON file named in PIVOT_FAKE_JEV ({"difficulty": 1.0,
    "lookup": 0.0, "select": "option0"} or {"error": "..."}), so the whole command can be
    exercised as a real process without a network."""

    def __init__(self, path: str) -> None:
        self.script = json.loads(Path(path).read_text(encoding="utf-8"))

    def ask(self, state_, questions: dict) -> dict:
        if self.script.get("error"):
            raise RuntimeError(str(self.script["error"]))
        answers = {}
        for name in questions:
            if name == "difficulty":
                answers[name] = {"type": "score", "score": float(self.script.get("difficulty", 2.0))}
            elif name == "lookup":
                answers[name] = {"type": "noul", "noul": float(self.script.get("lookup", 0.0))}
            else:
                answers[name] = {"type": "choice", "choice": str(self.script.get("select", "option0"))}
        return {"answers": answers, "usage": {"input_tokens": 100}, "model": "fake-jev"}


def make_jev(model: str, timeout: float):
    """-> a Jev client, or None with the reason. A missing key is not an error for `run`: it
    answers on the reference slot instead."""
    if os.environ.get("PIVOT_FAKE_JEV"):
        return executor.DirectJev(FakeJev(os.environ["PIVOT_FAKE_JEV"])), None
    try:
        from .jev import JevClient
        return executor.DirectJev(JevClient(model=model, timeout=timeout, retries=0, max_attempts=8)), None
    except Exception as error:      # noqa: BLE001 - a missing key and the like all mean "route without Jev"
        return None, f"{type(error).__name__}: {error}"


def read_prompt(options) -> str:
    if options.prompt_file:
        return Path(options.prompt_file).read_text(encoding="utf-8")
    words = [word for word in options.text if word != "-"]
    if words:
        return " ".join(words)
    if "-" in options.text or not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("no prompt: pass it as an argument, with --prompt-file, or on stdin")


def build_request(options) -> executor.Request:
    prompt = read_prompt(options)
    if not prompt.strip():
        raise ValueError("the prompt is empty")
    context = []
    if options.context_file:
        context = json.loads(Path(options.context_file).read_text(encoding="utf-8"))
        if not (isinstance(context, list) and all(isinstance(turn, dict) and "user" in turn for turn in context)):
            raise ValueError('--context-file must hold a JSON list of {"user": ..., "assistant": ...} turns')
    providers = tuple(part for part in options.providers.split(",") if part)
    if options.allow_billed:
        providers += tuple(name for name in ("openrouter", "ramp") if name not in providers)
    system = Path(options.system_file).read_text(encoding="utf-8") if options.system_file else options.system
    return executor.Request(
        prompt=prompt, system=system, context=context, budget_usd=options.budget_usd,
        max_seconds=options.max_seconds, call_timeout=options.call_timeout, latency_target=options.latency_target,
        providers=providers, models=tuple(options.models.split(",")) if options.models else None,
        route=options.route, slot=options.slot, difficulty=options.difficulty,
        check=executor.Check(options.expect, options.check_regex, options.check_cmd),
        max_attempts=options.max_attempts, billed_cap_usd=options.max_usd_billed,
        daily_billed_cap_usd=options.daily_usd_billed, dry_run=options.dry_run)


def emit(result: dict, as_json: bool) -> int:
    code = int(result.get("exit_code", 1))
    if as_json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    elif result.get("ok") and result["outcome"] != "planned":
        sys.stdout.write(result["text"].rstrip("\n") + "\n")
    elif result.get("ok"):
        sys.stdout.write(" -> ".join(result.get("plan", [])) + "\n")
    else:
        sys.stderr.write(f"pivot: {result['outcome']}: {result.get('detail', '')}\n")
    sys.stdout.flush()
    return code


def failure(outcome: str, detail: str, as_json: bool) -> int:
    return emit({"schema": executor.SCHEMA, "ok": False, "outcome": outcome, "detail": detail, "text": "",
                 "exit_code": executor.EXIT[outcome]}, as_json)


def on_signal(signum, _frame) -> None:
    # A caller's timeout or cancel: stop the CLI children now and leave. dsco gives 250 ms
    # between TERM and KILL, so nothing slower than this belongs here.
    slots_mod.terminate_children(signal.SIGTERM)
    os._exit(128 + signum)


def cmd_run(options) -> int:
    as_json = options.json or options.output_format == "json"
    try:
        return _run(options, as_json)
    except BrokenPipeError:             # the caller stopped reading
        return 1
    except Exception as error:          # noqa: BLE001 - a traceback in a captured stream helps no caller
        return failure("internal_error", f"{type(error).__name__}: {error}", as_json)


def _run(options, as_json: bool) -> int:
    try:
        request = build_request(options)
    except (ValueError, OSError) as error:
        return failure("usage", str(error), as_json)
    try:
        policy = load_policy(options.policy)
    except PolicyError as error:
        return failure("config_error", str(error), as_json)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, on_signal)
    if options.no_jev:
        jev, why = None, "disabled by --no-jev"
    elif request.route == "reference" or request.slot:
        jev, why = None, None
    else:
        jev, why = make_jev(options.jev_model, options.jev_timeout)
    log = (lambda line: print(f"[pivot] {line}", file=sys.stderr, flush=True)) if options.verbose else (lambda line: None)
    if jev is None and why:
        log(f"no Jev client: {why}")
    result = executor.execute(request, policy, jev=jev, jev_error=why, health=state.Health(),
                              ledger=state.BilledLedger(), prices=slots_mod.load_prices(), log=log)
    if not options.no_journal:
        try:
            state.append_jsonl(Path(options.journal) if options.journal else JOURNAL,
                               executor.journal_record(result))
        except OSError as error:        # a full disk must not turn an answer into a failure
            log(f"journal not written: {error}")
    return emit(result, as_json)


def cmd_fit(options) -> int:
    from .cli import make_jev as lab_jev, read_jsonl
    from .router import Grid
    source = RUNS / options.source
    if not (source / "grid.jsonl").exists():
        print(f"error: {source} has no grid.jsonl", file=sys.stderr)
        return 2
    items = json.loads((source / "items.json").read_text(encoding="utf-8"))
    rows = read_jsonl(source / "grid.jsonl")
    only = [part for part in (options.slots or "").split(",") if part]
    grid = Grid(items, rows, options.metric, slots_mod.load_prices(), only)
    if len(grid.items) < 10:
        print(f"error: only {len(grid.items)} complete problems; a policy needs at least 10", file=sys.stderr)
        return 2
    options.max_jev_requests = 2000
    data = compile_policy(grid, rows, lab_jev(options), source=options.source, time_value=options.time_value)
    if (source / "system.txt").exists():        # measured under custom instructions: `run` answers under them too
        data["system"] = (source / "system.txt").read_text(encoding="utf-8")
    out = Path(options.out) if options.out else DEFAULT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".tmp")
    partial.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    load_policy(partial)        # never install a policy the executor would refuse
    partial.replace(out)
    print(f"policy: {out}")
    print(f"reference: {data['reference']}  ({len(data['reliable'])} reliable slots, "
          f"{data['source']['slots']} measured on {data['source']['problems']} problems)")
    for bucket, entry in data["buckets"].items():
        print(f"difficulty {bucket} ({entry['problems']} problems): " + ", ".join(
            f"{o['slot']} {o['solved']}/{o['n']} {o['cost']:.4f}" for o in entry["options"]))
    return 0


def tool_version(name: str) -> dict:
    path = shutil.which(name)
    if not path:
        return {"found": False}
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15, check=False)
        return {"found": True, "path": path, "version": (done.stdout or done.stderr).strip().splitlines()[0][:80]}
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return {"found": True, "path": path, "version": None}


def cmd_doctor(options) -> int:
    report = {"schema": "jev-pivot.doctor/v1", "version": executor.VERSION, "python": sys.version.split()[0],
              "problems": [], "warnings": []}
    try:
        policy = load_policy(options.policy)
        report["policy"] = {"path": str(policy.path), "source": policy.source, "created": policy.created,
                            "age_days": round(policy.age_days(), 1), "reference": policy.reliable[0],
                            "providers": sorted({slot.provider for slot in policy.slots.values()})}
        if policy.age_days() > 30:
            report["warnings"].append("the policy is over 30 days old; models and prices move, refit it")
    except PolicyError as error:
        policy = None
        report["policy"] = {"error": str(error)}
        report["problems"].append(f"policy: {error}")
    report["tools"] = {name: tool_version(name) for name in ("claude", "codex", "dsco")}
    report["keys"] = {name: bool(os.environ.get(name)) for name in
                      ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "RAMP_ROUTER_API_KEY")}
    jev, why = make_jev(options.jev_model, options.jev_timeout)
    report["jev"] = {"client": jev is not None, "reason": why}
    if jev is None:
        report["warnings"].append(f"Jev is not usable ({why}); requests will run on the reference slot")
    try:
        state.STATE.mkdir(parents=True, exist_ok=True)
        probe = state.STATE / f".doctor.{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report["state"] = {"dir": str(state.STATE), "writable": True, "cooling_down": state.Health().snapshot(),
                           "billed_today_usd": round(state.BilledLedger().committed(), 4)}
    except OSError as error:
        report["state"] = {"dir": str(state.STATE), "writable": False}
        report["problems"].append(f"state directory is not writable: {error}")
    if policy:
        home = policy.slots[policy.reliable[0]].provider
        if home in ("claude", "codex") and not report["tools"][home]["found"]:
            report["problems"].append(f"the reference slot needs the `{home}` CLI, which is not on PATH")
    if options.probe and policy and not report["problems"]:
        request = executor.Request(prompt="What is 17 + 26?", check=executor.Check(expect="43"), max_seconds=120)
        result = executor.execute(request, policy, jev=jev, health=state.Health(), ledger=state.BilledLedger(),
                                  prices=slots_mod.load_prices())
        report["probe"] = {key: result.get(key) for key in ("ok", "outcome", "slot", "wall_s", "cost_usd", "detail")}
        report["probe"]["route"] = result["decision"]["route"]
        if not result["ok"]:
            report["problems"].append(f"probe failed: {result['outcome']}: {result.get('detail')}")
    report["ok"] = not report["problems"]
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_interop(options) -> int:
    pivot = str(ROOT / "pivot")
    run = [pivot, "run", "--json"] + (["--policy", str(Path(options.policy).resolve())] if options.policy else []) + ["-"]
    print(json.dumps({
        "schema": "jev-pivot.interop/v1", "version": executor.VERSION,
        "what": "text in, text out, tools off; the slot is picked per request from measured records",
        "dsco": ["dsco", "interop", "run", "generic", "--prompt", "<text>", "--stdin", "--json", "--"] + run,
        "direct": run,
        "prompt": "stdin (`-`), an argument, or --prompt-file",
        "result": "one JSON object on stdout (schema jev-pivot.run/v1); `text` is the answer",
        "exit_codes": {"0": "answered", "1": "run error (see outcome)", "2": "usage", "78": "config",
                       "143": "terminated"},
        "limits": {"default_deadline_s": 300, "billed_providers": "off unless --allow-billed"},
        "data_flow": "the prompt goes to the chosen model's provider; with a check it also goes to TypeSafe "
                     "Jev to pick the slot, unless --no-jev or --route reference"}, indent=2))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="pivot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("text", nargs="*", help="run: the prompt, or `-` to read it from stdin")
    parser.add_argument("--prompt-file")
    parser.add_argument("--system", help="run: system prompt (default: the one the policy was measured under, "
                                         "else: answer directly, no tools)")
    parser.add_argument("--system-file")
    parser.add_argument("--context-file", help='run: earlier turns as a JSON list of {"user", "assistant"}')
    parser.add_argument("--policy", help=f"policy file (default {DEFAULT_PATH})")
    parser.add_argument("--json", action="store_true", help="run: print one JSON result object")
    parser.add_argument("--output-format", default="text", choices=["text", "json"])
    parser.add_argument("--budget-usd", type=float, help="run: allowance at list price; Jev is told, and no "
                                                           "attempt starts once it is spent")
    parser.add_argument("--max-seconds", type=float, default=300.0, help="run: deadline for the whole run")
    parser.add_argument("--call-timeout", type=float, default=240.0)
    parser.add_argument("--latency-target", type=float, help="run: seconds per answer Jev is told to aim for")
    parser.add_argument("--providers", default="claude,codex",
                        help="run: providers that may be used (default: the subscription CLIs only)")
    parser.add_argument("--models", help="run: comma list of model names that may be used, e.g. to set a "
                                         "floor: claude-opus-5-5,claude-fable-5-2 (default: any in the policy)")
    parser.add_argument("--allow-billed", action="store_true",
                        help="run: also allow metered gateway slots in the policy (openrouter, ramp)")
    parser.add_argument("--max-usd-billed", type=float, default=0.05, help="run: billed cap for this run")
    parser.add_argument("--daily-usd-billed", type=float, default=1.0,
                        help="run: billed cap across every run on this machine today")
    parser.add_argument("--route", default="select", choices=["select", "derived", "reference"],
                        help="select: Jev picks from the policy's records; derived: arithmetic on the same "
                             "records; reference: always the cheapest fully reliable slot")
    parser.add_argument("--slot", help="run: pin one slot of the policy (fallbacks still apply)")
    parser.add_argument("--difficulty", type=float, help="run: 0..4 if the caller already knows; skips a Jev call")
    parser.add_argument("--expect", help="run: the exact final answer; a mismatch is retried on the reference")
    parser.add_argument("--check-regex", help="run: the answer must match this pattern")
    parser.add_argument("--check-cmd", help="run: command given the answer on stdin; exit 0 passes (no shell)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="run: decide and print the plan, call no slot")
    parser.add_argument("--journal", help=f"run: journal file (default {JOURNAL})")
    parser.add_argument("--no-journal", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="run: progress on stderr (not for dsco capture)")
    parser.add_argument("--no-jev", action="store_true",
                        help="run: never send the prompt to Jev; the difficulty comes from --difficulty, "
                             "or else the reference slot is used")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--jev-timeout", type=float, default=6.0)
    parser.add_argument("--from", dest="source", help="fit: the grid run to compile")
    parser.add_argument("--out", help="fit: where to write the policy")
    parser.add_argument("--slots", help="fit: comma list of slot-id globs to keep")
    parser.add_argument("--metric", default="usd_flat", choices=["usd", "usd_flat", "out_tokens", "wall_s"])
    parser.add_argument("--time-value", type=float, default=0.0)
    parser.add_argument("--probe", action="store_true", help="doctor: also answer one tiny real request")
    for stream in (sys.stdin, sys.stdout, sys.stderr):     # UTF-8 whatever the caller's locale
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        # Intermixed: before Python 3.12 plain parse_args gives a `*` positional nothing when options
        # come between the command and the prompt, and then rejects the prompt.
        options = parser.parse_intermixed_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    if options.command == "fit" and not options.source:
        print("error: fit needs --from RUN", file=sys.stderr)
        return 2
    return {"run": cmd_run, "fit": cmd_fit, "doctor": cmd_doctor, "interop": cmd_interop}[options.command](options)
