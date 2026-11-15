"""The sub-executor: one prompt in, one answer out, from the cheapest slot that should get it right.

Built to be fired off by another agent, so it never needs a human and never surprises one:

  * Jev reads the request and picks a slot from the compiled policy's measured records, knowing
    the budget, the deadline and the conversation. If Jev is unreachable the reference slot is
    used: the answer costs more, it does not fail.
  * A slot that errors, times out or returns nothing is replaced by the reference slot and then
    by reliable slots on another provider, and is put on a cooldown shared with other processes.
  * With a check (`expect`, a regex, or a command), Jev may try a cheaper slot first, and an
    answer that fails the check is retried on the reference slot.
  * Without a check a wrong answer would not be caught, so only a slot with a clean measured
    record is used. On the measured grid, letting the router gamble there saved 9% and got 2
    of 34 answers wrong. When one slot is the clean choice at every difficulty, Jev is not
    even asked.
  * Billed gateways are off unless allowed, and then capped per run and per day.
  * Every run ends inside its deadline with a result that says what happened and what it cost.

It answers from reasoning alone. Slots run with tools off and no file access; the measurements
behind the policy cover nothing else.
"""
from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field

from . import meta
from .policy import Policy
from .router import bucket_of, jev_state, profile_questions
from .slots import Slot, cost as receipt_cost, run_slot, usd, worst_case_usd
from .tasks import check as answer_matches, render

SCHEMA = "jev-pivot.run/v1"
VERSION = "1.0.0"
SYSTEM = ("Answer the user's request directly and accurately. You have no tools and cannot run code or "
          "read files; reason it through yourself. Reply with the answer only, without preamble.")
SUBSCRIPTIONS = ("claude", "codex")
MIN_CALL_S = 5.0        # no call is started with less time than this left
DEADLINE_MARGIN_S = 2.0

# outcome -> process exit code, following dsco's own headless convention (0 ok, 1 run error,
# 2 usage, 78 config, 143 SIGTERM).
EXIT = {"answered": 0, "verified": 0, "planned": 0, "check_failed": 1, "no_answer": 1,
        "budget_exhausted": 1, "deadline_exceeded": 1, "internal_error": 1, "usage": 2, "config_error": 78}
SLOW_S = 120.0          # a slot that used this much time and still timed out is held against it


@dataclass
class Check:
    """How an answer is verified, so a wrong one can be retried on a stronger slot."""
    expect: str | None = None       # the exact value of the final answer
    regex: str | None = None        # must match somewhere in the answer
    command: str | None = None      # argv (no shell); the answer on stdin; exit 0 passes
    timeout: float = 60.0

    def __bool__(self) -> bool:
        return any(value is not None for value in (self.expect, self.regex, self.command))

    def run(self, text: str) -> tuple[bool, str]:
        if self.expect is not None and not answer_matches(text, self.expect):
            return False, "answer is not the expected value"
        if self.regex is not None and not re.search(self.regex, text, re.MULTILINE):
            return False, "answer does not match the pattern"
        if self.command is not None:
            try:
                done = subprocess.run(shlex.split(self.command), input=text, text=True, capture_output=True,
                                      timeout=self.timeout, check=False, encoding="utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                return False, "check command timed out"
            except (OSError, ValueError) as error:
                return False, f"check command could not run: {error}"
            if done.returncode != 0:
                return False, f"check command exit {done.returncode}: {(done.stderr or done.stdout).strip()[:160]}"
        return True, "passed"


@dataclass
class Request:
    prompt: str
    system: str = SYSTEM
    context: list[dict] = field(default_factory=list)    # earlier turns: [{"user": ..., "assistant": ...}]
    budget_usd: float | None = None      # this run's allowance at list price; Jev is told what is left
    max_seconds: float = 300.0           # the whole run, every attempt included
    call_timeout: float = 240.0
    latency_target: float | None = None
    providers: tuple[str, ...] = SUBSCRIPTIONS
    models: tuple[str, ...] | None = None    # allow-list of model names; None allows every model
    route: str = "select"                # select (Jev picks) | derived (arithmetic) | reference
    slot: str | None = None              # pin one slot; fallbacks still apply if it fails
    difficulty: float | None = None      # 0..4 when the caller already knows; skips Jev's profile
    check: Check = field(default_factory=Check)
    max_attempts: int = 3
    billed_cap_usd: float = 0.05         # billed gateways: this run
    daily_billed_cap_usd: float = 1.0    # billed gateways: every run on this machine today
    dry_run: bool = False                # decide and report, call nothing


class DirectJev:
    """The interface `meta.ask` expects, over a plain client, with a usage tally."""

    def __init__(self, client) -> None:
        self.client, self.requests, self.input_tokens, self.model = client, 0, 0, None

    def ask(self, state, questions: dict) -> dict:
        reply = self.client.ask(state, questions)
        self.requests += 1
        self.input_tokens += int((reply.get("usage") or {}).get("input_tokens") or 0)
        self.model = reply.get("model") or self.model
        return reply["answers"]

    def summary(self) -> dict:
        return {"requests": self.requests, "input_tokens": self.input_tokens, "model": self.model,
                "usd": round(self.input_tokens * 0.042 / 1_000_000, 6)}


def _decide(request: Request, policy: Policy, jev, allowed, blocked, remaining_s: float,
            jev_error: str | None = None) -> dict:
    """Which slot to try first, and why. Never raises: every failure degrades to the reference slot."""
    reference = policy.reference(allowed, blocked)
    decision = {"route": request.route, "reference": reference, "pick": reference, "bucket": None,
                "difficulty": request.difficulty, "lookup": None, "offered": [], "fallback": None}
    if reference is None:
        decision.update(pick=None, fallback="no reliable slot is usable: providers disallowed or cooling down")
        return decision
    if request.slot:
        if request.slot in policy.slots and policy.usable(request.slot, allowed, blocked):
            decision.update(route="pinned", pick=request.slot)
        else:
            decision.update(route="pinned", fallback=f"pinned slot {request.slot} is not usable; using the reference")
        return decision
    if request.route == "reference":
        return decision
    item = {"context": request.context, "message": request.prompt, "prompt": render(request.context, request.prompt)}
    checked, allowance = bool(request.check), request.budget_usd
    choices = {bucket: policy.options(bucket, reference, allowed, blocked) for bucket in policy.buckets}

    def by_arithmetic(bucket: int) -> str:
        # Unchecked: only a clean record qualifies (risk 0). Checked: the lowest expected cost.
        return meta.derive(choices[bucket], reference, allowance=allowance, checked=checked, risk=0.0)

    if not checked and len({by_arithmetic(bucket) for bucket in choices}) == 1:
        pick = by_arithmetic(0)
        decision.update(route="clean-record", pick=pick, offered=[pick])
        return decision
    bucket = None if request.difficulty is None else max(0, min(4, round(request.difficulty)))
    if bucket is None:
        if jev is None:
            decision["fallback"] = f"no Jev client ({jev_error or 'none given'}); using the reference"
            return decision
        try:
            profile = jev.ask(jev_state(item), profile_questions())
            bucket = bucket_of(profile)
            decision.update(difficulty=profile["difficulty"]["score"], lookup=profile["lookup"]["noul"])
        except Exception as error:      # noqa: BLE001 - the router being down must not fail the run
            decision["fallback"] = f"Jev profile failed ({type(error).__name__}: {str(error)[:80]}); using the reference"
            return decision
    opts = choices[bucket]
    ref = next(option for option in opts if option["slot"] == reference)
    decision.update(bucket=bucket, offered=[option["slot"] for option in opts])
    if not checked:
        decision.update(route="clean-record", pick=by_arithmetic(bucket))
        return decision
    if request.route == "derived" or jev is None:
        decision.update(route="derived", pick=by_arithmetic(bucket))
        return decision
    budget = {"unit": "usd at list price", "reference_cost": round(ref["cost"], 6),
              "allowance": "no cap" if allowance is None else round(allowance, 6), "seconds_left": round(remaining_s)}
    if request.latency_target is not None:
        budget["latency_target_s"] = request.latency_target
    try:
        decision["pick"] = meta.ask(jev, item, bucket, opts, reference, budget=budget,
                                    context=meta.turn_context(item), checked=True)
    except Exception as error:      # noqa: BLE001
        decision.update(route="derived", pick=by_arithmetic(bucket),
                        fallback=f"Jev selection failed ({type(error).__name__}: {str(error)[:80]}); used arithmetic")
    return decision


def execute(request: Request, policy: Policy, *, jev=None, jev_error: str | None = None, health=None,
            ledger=None, prices=None, runner=run_slot, clock=time.monotonic, log=lambda line: None) -> dict:
    """Run one request to an answer or a clear failure. Never raises for a provider or router fault."""
    started = clock()
    deadline = started + request.max_seconds
    prices = prices or {}
    def allowed(slot: Slot) -> bool:
        return slot.provider in request.providers and (request.models is None or slot.model in request.models)

    blocked = (lambda slot_id: health.blocked(slot_id)) if health else (lambda slot_id: None)
    prompt = render(request.context, request.prompt)
    result = {"schema": SCHEMA, "version": VERSION, "id": uuid.uuid4().hex[:16], "ok": False,
              "outcome": "no_answer", "text": "",
              "slot": None, "verified": None, "attempts": [], "cost_usd": 0.0, "billed_usd": 0.0,
              "policy": {"source": policy.source.get("run"), "created": policy.created},
              "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "prompt_chars": len(prompt)}

    decision = _decide(request, policy, jev, allowed, blocked, deadline - clock(), jev_error)
    result["decision"] = decision
    log(f"decision: {decision['route']} -> {decision['pick']} (bucket {decision['bucket']}, "
        f"reference {decision['reference']}){' | ' + decision['fallback'] if decision['fallback'] else ''}")
    first = decision["pick"]
    if first is None:
        result.update(outcome="no_answer", detail=decision["fallback"])
        return _finish(result, jev, started, clock)
    chain = [first]
    if decision["reference"] and decision["reference"] != first:
        chain.append(decision["reference"])
    chain += [slot for slot in policy.fallbacks(decision["reference"] or first, allowed, blocked) if slot not in chain]
    result["plan"] = chain[:request.max_attempts]
    if request.dry_run:
        result.update(ok=True, outcome="planned")
        return _finish(result, jev, started, clock)

    best, detail, calls = None, "no slot produced an answer", 0
    for slot_id in chain:
        if calls >= request.max_attempts:        # slots skipped without a call do not use up attempts
            break
        remaining = deadline - clock()
        if remaining < MIN_CALL_S + DEADLINE_MARGIN_S:
            result["outcome"], detail = "deadline_exceeded", f"{remaining:.0f}s left of {request.max_seconds:.0f}s"
            break
        if request.budget_usd is not None and result["cost_usd"] >= request.budget_usd:
            result["outcome"], detail = "budget_exhausted", f"spent {result['cost_usd']:.4f} of {request.budget_usd:.4f}"
            break
        reason = blocked(slot_id)
        if reason:
            result["attempts"].append({"slot": slot_id, "status": "skipped", "error": reason})
            continue
        slot = policy.slots[slot_id]
        reservation, worst = None, worst_case_usd(slot, request.system, prompt)
        if slot.metered:
            over = (result["billed_usd"] + worst > request.billed_cap_usd
                    or (request.budget_usd is not None and result["cost_usd"] + worst > request.budget_usd))
            if not over and ledger is not None:
                reservation = ledger.reserve(slot.provider, worst, request.daily_billed_cap_usd)
                over = reservation is None
            if over:
                result["attempts"].append({"slot": slot_id, "status": "skipped",
                                           "error": f"billed cap: this call could cost {worst:.4f}"})
                continue
        allotted = max(MIN_CALL_S, min(request.call_timeout, remaining - DEADLINE_MARGIN_S))
        calls += 1
        receipt = runner(slot, prompt, system=request.system, timeout=allotted)
        attempt = _attempt(slot, receipt, prices)
        result["cost_usd"] += attempt["cost_usd"] or 0.0
        result["billed_usd"] += attempt["billed_usd"]
        if reservation is not None and ledger is not None:
            ledger.settle(reservation, slot.provider, attempt["billed_usd"])
        result["attempts"].append(attempt)
        answered = attempt["status"] == "ok"
        # A short timeout can be the caller's deadline or a hard problem, not the slot's fault.
        if health and (attempt["status"] != "timeout" or allotted >= SLOW_S):
            health.record(slot_id, answered or attempt["status"] == "empty", attempt.get("error"))
        log(f"attempt {len(result['attempts'])}: {slot_id} {attempt['status']} {attempt['wall_s']}s "
            f"${attempt['cost_usd'] or 0:.4f}{' ' + str(attempt.get('error'))[:100] if attempt.get('error') else ''}")
        if not answered:
            detail = f"{slot_id}: {attempt.get('error') or attempt['status']}"
            continue
        text = receipt["text"]
        if request.check:
            passed, why = request.check.run(text)
            attempt["check"] = "pass" if passed else "fail"
            if not passed:
                # Later slots in the chain are the reliable ones: keep the latest answer.
                best, detail = (slot_id, text), f"{slot_id}: {why}"
                result["outcome"] = "check_failed"
                continue
            result.update(ok=True, outcome="verified", verified=True, text=text, slot=slot_id)
        else:
            result.update(ok=True, outcome="answered", text=text, slot=slot_id)
        break
    if not result["ok"]:
        if result["outcome"] == "no_answer" and deadline - clock() < MIN_CALL_S + DEADLINE_MARGIN_S:
            result["outcome"] = "deadline_exceeded"      # it ended because the time ran out
        result["detail"] = detail
        if best:        # an answer that failed its check is still returned, marked unverified
            result.update(slot=best[0], text=best[1], verified=False)
    return _finish(result, jev, started, clock)


def _attempt(slot: Slot, receipt: dict, prices: dict) -> dict:
    status = "ok" if receipt.get("ok") else ("timeout" if receipt.get("error") == "timeout" else "error")
    if status == "ok" and not str(receipt.get("text") or "").strip():
        status = "empty"        # tokens spent, nothing said: never handed back as an answer
    try:
        listed = receipt_cost(receipt, "usd_flat", prices) if receipt.get("usage") else None
    except ValueError:          # a model with no list price: the cost as reported, if any
        listed = usd(receipt, prices)
    attempt = {"slot": slot.id, "provider": slot.provider, "status": status, "wall_s": receipt.get("wall_s", 0.0),
               "cost_usd": round(listed, 6) if listed is not None else None,
               "billed_usd": round(usd(receipt, prices) or 0.0, 6) if slot.metered else 0.0,
               "tokens": receipt.get("usage")}
    if status != "ok":
        attempt["error"] = str(receipt.get("error") or "empty answer")[:300]
    return attempt


def _finish(result: dict, jev, started: float, clock) -> dict:
    result["cost_usd"], result["billed_usd"] = round(result["cost_usd"], 6), round(result["billed_usd"], 6)
    result["wall_s"] = round(clock() - started, 2)
    result["jev"] = jev.summary() if hasattr(jev, "summary") else None
    result["exit_code"] = EXIT.get(result["outcome"], 1)
    return result


def journal_record(result: dict) -> dict:
    """What is kept about a run: everything but the answer and the prompt, which may be private."""
    return {**{key: value for key, value in result.items() if key != "text"},
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "answer_chars": len(result.get("text") or "")}
