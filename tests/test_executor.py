"""Offline tests for the sub-executor: no model call, no Jev call, no network.

The last class runs the real command as a child process, with a scripted provider and a
scripted Jev, because what a calling agent sees is bytes on stdout and an exit code.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pivotlab import executor, grid as grid_mod, router, state, tasks  # noqa: E402
from pivotlab.policy import SCHEMA, Policy, PolicyError, compile_policy, load_policy  # noqa: E402

CHEAP, REF, ALT, GATE = "tiny@low", "solid@high", "other@high", "openrouter:gw/cheap@low"


def option(slot, solved, n, p, cost, overall=(30, 34)):
    return {"slot": slot, "solved": solved, "n": n, "p": p, "cost": cost, "wall": 5.0, "price": cost,
            "overall_solved": overall[0], "overall_n": overall[1]}


def make_policy(**changes) -> Policy:
    top = (34, 34)
    data = {
        "schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"run": "unit", "problems": 34, "slots": 4, "metric": "usd_flat"},
        "reference": REF, "reliable": [REF, ALT],
        "slots": {
            CHEAP: {"provider": "claude", "model": "tiny", "effort": "low", "price": None,
                    "solved": 30, "n": 34, "cost": 0.001, "wall": 4.0},
            REF: {"provider": "claude", "model": "solid", "effort": "high", "price": None,
                  "solved": 34, "n": 34, "cost": 0.010, "wall": 6.0},
            ALT: {"provider": "codex", "model": "other", "effort": "high", "price": None,
                  "solved": 34, "n": 34, "cost": 0.020, "wall": 9.0},
            GATE: {"provider": "openrouter", "model": "gw/cheap", "effort": "low", "price": [1.0, 0.1, 2.0],
                   "solved": 33, "n": 34, "cost": 0.0005, "wall": 2.0}},
        "buckets": {str(bucket): {"problems": 7, "options": [option(REF, 7, 7, 1.0, 0.010, top)],
                                  "standby": [option(ALT, 7, 7, 1.0, 0.020, top)]} for bucket in range(5)}}
    data["buckets"]["1"]["options"] = [option(GATE, 7, 7, 0.99, 0.0005, (33, 34)), option(CHEAP, 7, 7, 0.98, 0.001),
                                       option(REF, 7, 7, 1.0, 0.010, top)]
    data["buckets"]["3"]["options"] = [option(CHEAP, 4, 7, 0.6, 0.002), option(REF, 7, 7, 1.0, 0.012, top)]
    data.update(changes)
    return Policy(data)


class StubJev:
    """Scripted judgments with the interface the executor uses."""

    def __init__(self, difficulty=1.0, select="option0", lookup=0.0, error=None):
        self.difficulty, self.select, self.lookup, self.error, self.asked = difficulty, select, lookup, error, []

    def ask(self, state_, questions):
        self.asked.append((state_, questions))
        if self.error:
            raise RuntimeError(self.error)
        if "select" in questions:
            labels = questions["select"]["criteria"]
            wanted = self.select if self.select in labels else next(
                key for key, text in labels.items() if self.select in text)
            return {"select": {"type": "choice", "choice": wanted}}
        return {"difficulty": {"type": "score", "score": self.difficulty},
                "lookup": {"type": "noul", "noul": self.lookup}}

    def summary(self):
        return {"requests": len(self.asked)}


def receipt(text="ANSWER: 43", ok=True, error=None, usd=0.002, **extra):
    return {"ok": ok, "text": text if ok else "", "error": error, "usd_reported": usd if ok else None,
            "usage": {"input": 100, "cached_input": 0, "output": 50, "reasoning": 40}, "wall_s": 1.5,
            "tool_events": 0, **extra}


class Runner:
    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, slot, prompt, *, system, timeout):
        self.calls.append({"slot": slot.id, "prompt": prompt, "system": system, "timeout": timeout})
        step = self.script[slot.id]
        if isinstance(step, list):
            step = step.pop(0) if len(step) > 1 else step[0]
        return {"slot": slot.id, "provider": slot.provider, "model": slot.model, "effort": slot.effort,
                **(step() if callable(step) else step)}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.health = state.Health(Path(self.tmp.name) / "health.json")
        self.ledger = state.BilledLedger(Path(self.tmp.name) / "billed.jsonl")
        self.policy = make_policy()

    def tearDown(self):
        self.tmp.cleanup()

    def run_request(self, runner, jev=None, **fields):
        request = executor.Request(prompt=fields.pop("prompt", "What is 17 + 26?"), **fields)
        return executor.execute(request, self.policy, jev=jev, health=self.health, ledger=self.ledger,
                                prices={}, runner=runner)

    def test_jev_picks_the_cheap_slot_and_its_answer_comes_back(self):
        runner, jev = Runner({CHEAP: receipt()}), StubJev(difficulty=1.0, select=CHEAP)
        result = self.run_request(runner, jev, check=executor.Check(regex="43"))
        self.assertEqual((result["ok"], result["outcome"], result["slot"], result["exit_code"]),
                         (True, "verified", CHEAP, 0))
        self.assertEqual(result["text"], "ANSWER: 43")
        self.assertEqual([call["slot"] for call in runner.calls], [CHEAP])
        self.assertEqual(result["decision"]["bucket"], 1)
        self.assertNotIn(GATE, result["decision"]["offered"])            # billed gateways are off by default
        self.assertEqual(result["decision"]["offered"], [CHEAP, REF])
        self.assertEqual(len(jev.asked), 2)                              # one profile, one selection
        told = jev.asked[1][0]
        self.assertEqual(told["budget"]["allowance"], "no cap")
        self.assertIn("seconds_left", told["budget"])
        self.assertIn("caught by a check", jev.asked[1][1]["select"]["instructions"])
        self.assertAlmostEqual(result["cost_usd"], 0.002)
        self.assertEqual(result["billed_usd"], 0.0)

    def test_without_a_check_only_a_clean_record_is_used_and_jev_is_not_asked(self):
        runner, jev = Runner({REF: receipt()}), StubJev(difficulty=1.0, select=CHEAP)
        result = self.run_request(runner, jev)
        self.assertEqual((result["outcome"], result["slot"], result["decision"]["route"]),
                         ("answered", REF, "clean-record"))
        self.assertEqual(jev.asked, [])                                  # one slot is clean at every difficulty
        clean = make_policy()
        clean.buckets[1][1]["p"] = 1.0                                   # tiny has a clean record on easy turns
        request = executor.Request(prompt="q")
        easy = executor.execute(request, clean, jev=StubJev(difficulty=1.0), prices={},
                                runner=Runner({CHEAP: receipt()}))
        self.assertEqual((easy["slot"], easy["decision"]["bucket"]), (CHEAP, 1))    # Jev only said how hard it is
        hard_jev = StubJev(difficulty=3.0)
        hard = executor.execute(request, clean, jev=hard_jev, prices={}, runner=Runner({REF: receipt()}))
        self.assertEqual((hard["slot"], len(hard_jev.asked)), (REF, 1))
        offline = executor.execute(request, clean, jev=StubJev(error="down"), prices={},
                                   runner=Runner({REF: receipt()}))
        self.assertEqual(offline["slot"], REF)                           # no judgment: the reference
        capped = executor.execute(executor.Request(prompt="q", budget_usd=0.005, difficulty=3.0), clean, prices={},
                                  runner=Runner({CHEAP: receipt()}))
        self.assertEqual(capped["slot"], CHEAP)                          # the caller's cap rules out the reference

    def test_the_router_being_down_costs_money_not_the_answer(self):
        runner, check = Runner({REF: receipt()}), executor.Check(regex="43")
        result = self.run_request(runner, StubJev(error="jev_http_503"), check=check)
        self.assertEqual((result["ok"], result["slot"]), (True, REF))
        self.assertIn("Jev profile failed", result["decision"]["fallback"])
        none = self.run_request(Runner({REF: receipt()}), None, check=check)
        self.assertEqual((none["ok"], none["slot"]), (True, REF))
        self.assertIn("no Jev client", none["decision"]["fallback"])

        class HalfJev(StubJev):                                          # profile works, selection fails
            def ask(self, state_, questions):
                if "select" in questions:
                    raise RuntimeError("timeout")
                return super().ask(state_, questions)

        half = self.run_request(Runner({REF: receipt(), CHEAP: receipt()}), HalfJev(difficulty=1.0), check=check)
        self.assertTrue(half["ok"])
        self.assertEqual(half["decision"]["route"], "derived")
        self.assertIn("used arithmetic", half["decision"]["fallback"])

    def test_a_failing_slot_falls_back_to_the_reference_then_to_another_provider(self):
        down = receipt(ok=False, error="exit 1: not logged in")
        runner, check = Runner({CHEAP: down, REF: down, ALT: receipt()}), executor.Check(regex="43")
        result = self.run_request(runner, StubJev(select=CHEAP), check=check)
        self.assertEqual((result["ok"], result["slot"]), (True, ALT))
        self.assertEqual([a["status"] for a in result["attempts"]], ["error", "error", "ok"])
        self.assertEqual(result["plan"], [CHEAP, REF, ALT])
        self.assertIsNotNone(self.health.blocked(CHEAP))                 # both now cool down for other processes
        self.assertIsNotNone(self.health.blocked(REF))
        again = self.run_request(Runner({ALT: receipt()}), StubJev(select=CHEAP), check=check)
        self.assertEqual((again["slot"], again["decision"]["reference"]), (ALT, ALT))   # no call wasted on them
        self.assertEqual(len(again["attempts"]), 1)

    def test_no_slot_answering_is_a_clear_failure_not_an_exception(self):
        down = receipt(ok=False, error="HTTP 500")
        result = self.run_request(Runner({CHEAP: down, REF: down, ALT: down}), StubJev(select=CHEAP),
                                  check=executor.Check(regex="43"))
        self.assertEqual((result["ok"], result["outcome"], result["exit_code"], result["text"]),
                         (False, "no_answer", 1, ""))
        self.assertIn("HTTP 500", result["detail"])
        self.assertEqual(len(result["attempts"]), 3)

    def test_an_empty_answer_is_never_handed_back(self):
        runner = Runner({CHEAP: receipt(text="   "), REF: receipt()})
        result = self.run_request(runner, StubJev(select=CHEAP), check=executor.Check(regex="43"))
        self.assertEqual((result["slot"], result["attempts"][0]["status"]), (REF, "empty"))
        self.assertIsNone(self.health.blocked(CHEAP))                    # it answered nothing, but it is not down

    def test_a_failed_check_is_retried_on_the_reference(self):
        runner = Runner({CHEAP: receipt("ANSWER: 44"), REF: receipt("ANSWER: 43")})
        jev = StubJev(select=CHEAP)
        result = self.run_request(runner, jev, check=executor.Check(expect="43"))
        self.assertEqual((result["outcome"], result["verified"], result["slot"], result["text"]),
                         ("verified", True, REF, "ANSWER: 43"))
        self.assertEqual([a.get("check") for a in result["attempts"]], ["fail", "pass"])
        self.assertIn("caught by a check", jev.asked[1][1]["select"]["instructions"])
        wrong = Runner({CHEAP: receipt("ANSWER: 1"), REF: receipt("ANSWER: 2"), ALT: receipt("ANSWER: 3")})
        failed = self.run_request(wrong, StubJev(select=CHEAP), check=executor.Check(expect="43"))
        self.assertEqual((failed["ok"], failed["outcome"], failed["exit_code"], failed["verified"]),
                         (False, "check_failed", 1, False))
        self.assertEqual((failed["slot"], failed["text"]), (ALT, "ANSWER: 3"))   # the last, most reliable try

    def test_checks_by_pattern_and_by_command(self):
        self.assertEqual(executor.Check(regex=r"^ANSWER: \d+$").run("ANSWER: 43")[0], True)
        self.assertEqual(executor.Check(regex=r"^ANSWER: \d+$").run("maybe 43")[0], False)
        passes = executor.Check(command=f"{sys.executable} -c 'import sys; sys.exit(0 if \"43\" in sys.stdin.read() else 3)'")
        self.assertEqual(passes.run("ANSWER: 43"), (True, "passed"))
        self.assertIn("exit 3", passes.run("ANSWER: 44")[1])
        self.assertIn("could not run", executor.Check(command="/nonexistent/checker").run("x")[1])
        self.assertFalse(executor.Check())
        self.assertTrue(executor.Check(expect="1"))

    def test_the_run_ends_inside_its_deadline(self):
        clock = Clock()

        def slow():
            clock.now += 50.0
            return receipt(ok=False, error="timeout")

        runner = Runner({CHEAP: slow, REF: slow, ALT: slow})
        request = executor.Request(prompt="q", max_seconds=60.0, call_timeout=240.0)
        result = executor.execute(request, self.policy, jev=StubJev(select=CHEAP), health=self.health,
                                  prices={}, runner=runner, clock=clock)
        self.assertEqual((result["outcome"], result["exit_code"]), ("deadline_exceeded", 1))
        self.assertEqual(len(runner.calls), 2)                           # the third attempt had under 7 s left
        self.assertEqual(runner.calls[0]["timeout"], 58.0)               # the whole budget minus the margin
        self.assertEqual(runner.calls[1]["timeout"], 8.0)
        self.assertLessEqual(result["wall_s"], 110.0)
        self.assertIsNone(self.health.blocked(CHEAP))                    # a timeout is not held against a slot

    def test_a_spent_budget_stops_further_attempts(self):
        runner = Runner({CHEAP: receipt("ANSWER: 1", usd=0.004), REF: receipt("ANSWER: 43")})
        jev = StubJev(select=CHEAP)
        result = self.run_request(runner, jev, budget_usd=0.003, check=executor.Check(expect="43"))
        self.assertEqual((result["outcome"], len(runner.calls)), ("budget_exhausted", 1))
        self.assertEqual((result["verified"], result["text"]), (False, "ANSWER: 1"))
        self.assertEqual(jev.asked[1][0]["budget"]["allowance"], 0.003)

    def test_billed_gateways_need_permission_and_stay_under_both_caps(self):
        both = ("claude", "codex", "openrouter")
        gate = receipt(usd=None, price={"input": 1.0, "cached_input": 0.1, "output": 2.0})
        check = executor.Check(regex="43")
        result = self.run_request(Runner({GATE: gate}), StubJev(select=GATE), providers=both, billed_cap_usd=0.10,
                                  check=check)
        self.assertEqual((result["ok"], result["slot"]), (True, GATE))
        self.assertAlmostEqual(result["billed_usd"], (100 * 1.0 + 50 * 2.0) / 1e6)
        self.assertAlmostEqual(self.ledger.committed("openrouter"), result["billed_usd"])   # reserved, then settled
        tight = self.run_request(Runner({GATE: gate, REF: receipt()}), StubJev(select=GATE), providers=both,
                                 billed_cap_usd=0.001, check=check)      # worst case is 4 cents: not sent
        self.assertEqual((tight["slot"], tight["attempts"][0]["status"]), (REF, "skipped"))
        self.assertIn("billed cap", tight["attempts"][0]["error"])
        daily = self.run_request(Runner({GATE: gate, REF: receipt()}), StubJev(select=GATE), providers=both,
                                 billed_cap_usd=0.10, daily_billed_cap_usd=0.01, check=check)
        self.assertEqual(daily["slot"], REF)

    def test_pins_floors_known_difficulty_and_dry_runs(self):
        pinned = self.run_request(Runner({ALT: receipt()}), None, slot=ALT)
        self.assertEqual((pinned["slot"], pinned["decision"]["route"]), (ALT, "pinned"))
        unknown = self.run_request(Runner({REF: receipt()}), None, slot="nope@max")
        self.assertEqual(unknown["slot"], REF)
        self.assertIn("not usable", unknown["decision"]["fallback"])
        floor = self.run_request(Runner({ALT: receipt()}), StubJev(select=CHEAP), models=("other",))
        self.assertEqual((floor["slot"], floor["decision"]["offered"]), (ALT, [ALT]))
        jev = StubJev(select=CHEAP)
        known = self.run_request(Runner({CHEAP: receipt()}), jev, difficulty=1.2, check=executor.Check(regex="43"))
        self.assertEqual((known["decision"]["bucket"], len(jev.asked)), (1, 1))     # no profile request
        hard = self.run_request(Runner({REF: receipt()}), None, difficulty=3.0, route="derived")
        self.assertEqual(hard["slot"], REF)                              # tiny failed 3 of 7 at this difficulty
        easy = self.run_request(Runner({REF: receipt()}), None, difficulty=1.0, route="derived")
        self.assertEqual(easy["slot"], REF)                              # unchecked: 0.98 is not a clean record
        checked = self.run_request(Runner({CHEAP: receipt()}), None, difficulty=1.0, route="derived",
                                   check=executor.Check(regex="43"))
        self.assertEqual(checked["slot"], CHEAP)                         # with a check the cheap try is worth it
        runner = Runner({})
        plan = self.run_request(runner, StubJev(select=CHEAP), dry_run=True, check=executor.Check(regex="43"))
        self.assertEqual((plan["outcome"], plan["plan"], runner.calls), ("planned", [CHEAP, REF, ALT], []))
        nothing = self.run_request(Runner({}), None, providers=("nobody",))
        self.assertEqual((nothing["ok"], nothing["outcome"]), (False, "no_answer"))

    def test_slots_skipped_without_a_call_do_not_use_up_attempts(self):
        self.health.record(CHEAP, False, "HTTP 429")
        policy = make_policy(reliable=[REF, ALT])
        down = receipt(ok=False, error="HTTP 500")
        runner = Runner({REF: down, ALT: receipt()})
        request = executor.Request(prompt="q", slot=CHEAP, max_attempts=2)       # the pin is cooling down
        result = executor.execute(request, policy, health=self.health, prices={}, runner=runner)
        self.assertEqual((result["ok"], result["slot"]), (True, ALT))
        self.assertEqual([a["status"] for a in result["attempts"]], ["error", "ok"])
        self.health.record(REF, True)
        self.health.record(ALT, False, "down")                           # cooling between the plan and the call
        late = executor.execute(executor.Request(prompt="q", max_attempts=1), policy, prices={}, runner=Runner({
            REF: down}), health=state.Health(Path(self.tmp.name) / "other.json"))
        self.assertEqual(len(late["attempts"]), 1)

    def test_a_long_timeout_is_held_against_a_slot_but_a_short_one_is_not(self):
        timed_out = receipt(ok=False, error="timeout")
        short = executor.Request(prompt="q", max_seconds=30.0, max_attempts=1)
        executor.execute(short, self.policy, health=self.health, prices={}, runner=Runner({REF: timed_out}))
        self.assertIsNone(self.health.blocked(REF))
        long = executor.Request(prompt="q", max_seconds=300.0, call_timeout=240.0, max_attempts=1)
        executor.execute(long, self.policy, health=self.health, prices={}, runner=Runner({REF: timed_out}))
        self.assertIn("timeout", self.health.blocked(REF))

    def test_the_conversation_reaches_both_jev_and_the_slot(self):
        runner, jev = Runner({CHEAP: receipt()}), StubJev(select=CHEAP)
        context = [{"user": "The list is [3, 9, 4].", "assistant": "Understood."}]
        self.run_request(runner, jev, prompt="How many items?", context=context, system="Be brief.",
                         check=executor.Check(regex="43"))
        self.assertIn("The list is [3, 9, 4].", runner.calls[0]["prompt"])
        self.assertTrue(runner.calls[0]["prompt"].endswith("How many items?"))
        self.assertEqual(runner.calls[0]["system"], "Be brief.")
        self.assertEqual(jev.asked[0][0]["conversation"], context)
        self.assertEqual(jev.asked[1][0]["context"]["prior_turns"], 1)

    def test_answers_come_under_the_instructions_the_policy_was_measured_with(self):
        measured = make_policy(system="Reply with one word.")
        runner = Runner({REF: receipt()})
        executor.execute(executor.Request(prompt="q"), measured, prices={}, runner=runner)
        self.assertEqual(runner.calls[0]["system"], "Reply with one word.")
        executor.execute(executor.Request(prompt="q", system="Be terse."), measured, prices={}, runner=runner)
        self.assertEqual(runner.calls[1]["system"], "Be terse.")                 # the caller's word wins
        executor.execute(executor.Request(prompt="q"), self.policy, prices={}, runner=runner)
        self.assertEqual(runner.calls[2]["system"], executor.SYSTEM)

    def test_the_journal_keeps_the_receipt_but_not_the_words(self):
        result = self.run_request(Runner({CHEAP: receipt("the secret answer")}), StubJev(select=CHEAP),
                                  prompt="a private question", check=executor.Check(regex="secret"))
        kept = executor.journal_record(result)
        self.assertNotIn("text", kept)
        self.assertNotIn("private question", json.dumps(kept))
        self.assertNotIn("secret answer", json.dumps(kept))
        self.assertEqual((kept["answer_chars"], kept["slot"], len(kept["prompt_sha256"])), (17, CHEAP, 64))


class StateTests(unittest.TestCase):
    def test_health_backs_off_and_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock()
            health = state.Health(Path(tmp) / "h.json", clock)
            self.assertIsNone(health.blocked("s"))
            health.record("s", False, "HTTP 429")
            self.assertIn("HTTP 429", health.blocked("s"))
            clock.now += 31
            self.assertIsNone(health.blocked("s"))                       # 30 s after the first error
            health.record("s", False, "HTTP 429")
            clock.now += 31
            self.assertIsNotNone(health.blocked("s"))                    # 60 s after the second
            health.record("s", True)
            self.assertIsNone(health.blocked("s"))
            self.assertEqual(health.snapshot(), {})
            for _ in range(12):
                health.record("s", False, "down")
            clock.now += state.Health.MAX_S + 1
            self.assertIsNone(health.blocked("s"))                       # the cooldown is capped

    def test_the_daily_billed_cap_holds_across_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock()
            ledger = state.BilledLedger(Path(tmp) / "b.jsonl", clock)
            first = ledger.reserve("openrouter", 0.4, cap=1.0)
            second = ledger.reserve("openrouter", 0.4, cap=1.0)
            self.assertTrue(first and second)
            self.assertIsNone(ledger.reserve("ramp", 0.4, cap=1.0))      # the cap is for all gateways together
            ledger.settle(first, "openrouter", 0.01)
            self.assertAlmostEqual(ledger.committed(), 0.41)
            self.assertTrue(ledger.reserve("ramp", 0.4, cap=1.0))
            clock.now += state.BilledLedger.STALE_S + 1                  # a killed process never settles
            self.assertAlmostEqual(ledger.committed(), 0.01)
            clock.now += 86400
            self.assertEqual(ledger.committed(), 0.0)                    # a new day

    def test_concurrent_writers_never_tear_the_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            script = ("import sys; sys.path[:0]=[%r]; from pivotlab import state; from pathlib import Path\n"
                      "for n in range(150): state.append_jsonl(Path(%r), {'writer': sys.argv[1], 'n': n, "
                      "'pad': 'x' * 300})" % (str(ROOT), str(path)))
            writers = [subprocess.Popen([sys.executable, "-c", script, str(w)]) for w in range(4)]
            self.assertEqual([w.wait(timeout=60) for w in writers], [0, 0, 0, 0])
            rows = state.read_jsonl(path)
            self.assertEqual(len(rows), 600)
            self.assertEqual(len(path.read_text().splitlines()), 600)
            path.write_text(path.read_text() + '{"torn": tr')            # a writer caught mid-line
            self.assertEqual(len(state.read_jsonl(path)), 600)

    def test_the_cap_holds_when_many_processes_reserve_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = state.BilledLedger(Path(tmp) / "b.jsonl")
            got = []
            threads = [threading.Thread(target=lambda: got.append(ledger.reserve("openrouter", 0.3, cap=1.0)))
                       for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sum(1 for reservation in got if reservation), 3)


class PolicyTests(unittest.TestCase):
    def test_a_policy_compiles_from_a_grid_and_round_trips(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_offline as lab
        with tempfile.TemporaryDirectory() as tmp:
            rows, _, _ = lab.make_grid(tmp)
            grid = router.Grid(lab.SUITE, rows, "usd", {})
            jev = router.CachedJev(lab.FakeJev(), Path(tmp) / "jev.jsonl")
            data = compile_policy(grid, rows, jev, source="unit")
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(data))
            policy = load_policy(path)
            self.assertEqual((policy.reliable, data["reference"]), ([lab.C.id], lab.C.id))
            self.assertEqual(policy.slots[lab.A.id].provider, "claude")
            everyone, nobody = (lambda slot: True), (lambda slot_id: None)
            easy = policy.options(0, lab.C.id, everyone, nobody)
            self.assertEqual([o["slot"] for o in easy], [lab.A.id, lab.B.id, lab.C.id])
            self.assertEqual((easy[0]["solved"], easy[0]["n"]), (4, 4))
            claude_only = policy.options(0, lab.C.id, lambda slot: slot.provider == "codex", nobody)
            self.assertEqual([o["slot"] for o in claude_only], [lab.C.id])
            self.assertLess(policy.age_days(), 0.01)
            self.assertEqual(data["source"]["problems"], len(lab.SUITE))

    def test_reference_and_fallbacks_respect_allow_lists_and_cooldowns(self):
        policy = make_policy()
        everyone, nobody = (lambda slot: True), (lambda slot_id: None)
        self.assertEqual(policy.reference(everyone, nobody), REF)
        self.assertEqual(policy.reference(everyone, lambda slot_id: "down" if slot_id == REF else None), ALT)
        self.assertEqual(policy.reference(lambda slot: slot.provider == "codex", nobody), ALT)
        self.assertIsNone(policy.reference(lambda slot: False, nobody))
        self.assertEqual(policy.fallbacks(REF, everyone, nobody), [ALT])
        standby = policy.options(2, ALT, lambda slot: slot.provider == "codex", nobody)
        self.assertEqual([(o["slot"], o["n"]) for o in standby], [(ALT, 7)])      # its own record, from standby

    def test_a_bad_policy_is_refused_with_a_reason(self):
        with self.assertRaises(PolicyError):
            Policy({"schema": "something-else"})
        with self.assertRaises(PolicyError):
            make_policy(reliable=["ghost@max"])
        with self.assertRaises(PolicyError):
            make_policy(reliable=[])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PolicyError, "pivot fit"):
                load_policy(Path(tmp) / "missing.json")
            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json")
            with self.assertRaisesRegex(PolicyError, "not valid JSON"):
                load_policy(broken)


def fake_policy() -> dict:
    """A policy over the scripted `fake` provider, for running the real command."""
    slots = {"weak@low": {"provider": "fake", "model": "weak", "effort": "low", "price": None,
                          "solved": 30, "n": 34, "cost": 0.001, "wall": 1.0},
             "strong@high": {"provider": "fake", "model": "strong", "effort": "high", "price": None,
                             "solved": 34, "n": 34, "cost": 0.010, "wall": 1.0}}
    options = [option("weak@low", 7, 7, 0.98, 0.001), option("strong@high", 7, 7, 1.0, 0.010, (34, 34))]
    return {"schema": SCHEMA, "created": "2026-11-14T00:00:00Z", "source": {"run": "fake"},
            "reference": "strong@high", "reliable": ["strong@high"], "slots": slots,
            "buckets": {str(b): {"problems": 7, "options": options, "standby": []} for b in range(5)}}


class CommandTests(unittest.TestCase):
    """The command exactly as another agent spawns it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "policy.json").write_text(json.dumps(fake_policy()))
        self.script({"weak@low": {"text": "ANSWER: 43"}, "strong@high": {"text": "ANSWER: 43"}})
        self.jev({"difficulty": 1.0, "select": "option0"})
        self.env = {**os.environ, "PYTHONPATH": str(ROOT),
                    "PIVOT_STATE": str(self.dir / "state"), "PIVOT_FAKE_SCRIPT": str(self.dir / "script.json"),
                    "PIVOT_FAKE_JEV": str(self.dir / "jev.json")}

    def tearDown(self):
        self.tmp.cleanup()

    def script(self, acts):
        (self.dir / "script.json").write_text(json.dumps(acts))

    def jev(self, answers):
        (self.dir / "jev.json").write_text(json.dumps(answers))

    def argv(self, *extra):
        return [sys.executable, "-m", "pivotlab.cli", "run", "--policy", str(self.dir / "policy.json"),
                "--providers", "fake", *extra]

    def run_cli(self, *extra, stdin="What is 17 + 26?"):
        return subprocess.run(self.argv(*extra), input=stdin, capture_output=True, text=True, env=self.env,
                              timeout=60, cwd="/")

    def test_json_mode_prints_exactly_one_object_and_nothing_else(self):
        done = self.run_cli("--json", "--check-regex", "43", "-")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        self.assertEqual(done.stdout.count("\n"), 1)
        result = json.loads(done.stdout)
        self.assertEqual((result["schema"], result["ok"], result["text"], result["slot"]),
                         (executor.SCHEMA, True, "ANSWER: 43", "weak@low"))
        self.assertEqual(result["decision"]["route"], "select")
        journal = state.read_jsonl(self.dir / "state" / "executions.jsonl")
        self.assertEqual((len(journal), journal[0]["id"]), (1, result["id"]))
        self.assertNotIn("text", journal[0])

    def test_text_mode_prints_only_the_answer(self):
        done = self.run_cli()
        self.assertEqual((done.returncode, done.stdout, done.stderr), (0, "ANSWER: 43\n", ""))
        as_argument = self.run_cli("What", "is", "17", "+", "26?", stdin="")
        self.assertEqual((as_argument.returncode, as_argument.stdout), (0, "ANSWER: 43\n"))
        prompt = self.dir / "prompt.txt"
        prompt.write_text("from a file")
        self.script({"*": {"text": "echo: {prompt}"}})
        from_file = self.run_cli("--prompt-file", str(prompt), stdin="")
        self.assertEqual(from_file.stdout, "echo: from a file\n")

    def test_failures_have_exit_codes_and_one_json_object(self):
        missing = subprocess.run([sys.executable, "-m", "pivotlab.cli", "run", "--json", "--policy",
                                  str(self.dir / "nope.json"), "-"], input="q", capture_output=True, text=True,
                                 env=self.env, timeout=60)
        self.assertEqual((missing.returncode, json.loads(missing.stdout)["outcome"]), (78, "config_error"))
        empty = self.run_cli("--json", "-", stdin="   ")
        self.assertEqual((empty.returncode, json.loads(empty.stdout)["outcome"]), (2, "usage"))
        self.script({"*": {"error": "HTTP 503: upstream down"}})
        down = self.run_cli("--json", "-")
        result = json.loads(down.stdout)
        self.assertEqual((down.returncode, result["outcome"], result["ok"], down.stderr), (1, "no_answer", False, ""))
        self.assertIn("HTTP 503", result["detail"])
        plain = self.run_cli()
        self.assertEqual((plain.returncode, plain.stdout), (1, ""))
        self.assertIn("pivot: no_answer", plain.stderr)
        bad_flag = subprocess.run(self.argv("--route", "nonsense"), input="q", capture_output=True, text=True,
                                  env=self.env, timeout=60)
        self.assertEqual(bad_flag.returncode, 2)

    def test_checks_escalate_and_jev_outages_degrade_in_the_real_command(self):
        self.script({"weak@low": {"text": "ANSWER: 44"}, "strong@high": {"text": "ANSWER: 43"}})
        result = json.loads(self.run_cli("--json", "--expect", "43", "-").stdout)
        self.assertEqual((result["outcome"], result["slot"], len(result["attempts"])), ("verified", "strong@high", 2))
        self.jev({"error": "jev_http_503"})
        result = json.loads(self.run_cli("--json", "--check-regex", "43", "-").stdout)
        self.assertEqual((result["ok"], result["slot"]), (True, "strong@high"))
        self.assertIn("Jev profile failed", result["decision"]["fallback"])
        plan = json.loads(self.run_cli("--json", "--dry-run", "--route", "reference", "-").stdout)
        self.assertEqual((plan["outcome"], plan["plan"], plan["attempts"]), ("planned", ["strong@high"], []))

    def test_a_slot_that_fails_cools_down_for_the_next_process(self):
        self.script({"weak@low": {"error": "HTTP 429"}, "strong@high": {"text": "ANSWER: 43"}})
        first = json.loads(self.run_cli("--json", "--check-regex", "43", "-").stdout)
        self.assertEqual([a["status"] for a in first["attempts"]], ["error", "ok"])
        second = json.loads(self.run_cli("--json", "--check-regex", "43", "-").stdout)
        self.assertEqual([a["slot"] for a in second["attempts"]], ["strong@high"])     # weak@low was not retried

    def test_sigterm_stops_the_child_and_exits_at_once(self):
        marker = self.dir / "child.pid"
        child = f"import os, time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(120)"
        self.script({"*": {"exec": [sys.executable, "-c", child]}})
        proc = subprocess.Popen(self.argv("--json", "--route", "reference", "-"), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)
        proc.stdin.write("q")
        proc.stdin.close()
        deadline = time.time() + 20
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(marker.exists(), "the slot's child never started")
        time.sleep(0.2)
        child_pid = int(marker.read_text())
        sent = time.time()
        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(timeout=5), 143)
        # dsco sends KILL 250 ms after TERM. Shared CI runners are too jittery to hold that bound.
        self.assertLess(time.time() - sent, 2.0 if os.environ.get("CI") else 0.25)
        proc.stdout.close()
        proc.stderr.close()
        for _ in range(40):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(child_pid, signal.SIGKILL)
            self.fail("the slot's child outlived the executor")

    def test_a_slow_slot_cannot_outlast_the_deadline(self):
        self.script({"*": {"exec": [sys.executable, "-c", "import time; time.sleep(120)"]}})
        started = time.time()
        done = self.run_cli("--json", "--route", "reference", "--max-seconds", "8", "-")
        result = json.loads(done.stdout)
        self.assertEqual((done.returncode, result["outcome"]), (1, "deadline_exceeded"))
        self.assertLess(time.time() - started, 14)
        self.assertEqual(result["attempts"][0]["status"], "timeout")

    def test_answers_survive_a_bare_locale(self):
        self.script({"*": {"text": "café – 43 ✓ 東京"}})
        env = {**self.env, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0"}
        done = subprocess.run(self.argv("--json", "-"), input="Où est le café ?".encode("utf-8"),
                              capture_output=True, env=env, timeout=60)
        self.assertEqual((done.returncode, done.stderr), (0, b""))
        self.assertEqual(json.loads(done.stdout.decode("utf-8"))["text"], "café – 43 ✓ 東京")
        plain = subprocess.run(self.argv("-"), input=b"q", capture_output=True, env=env, timeout=60)
        self.assertEqual(plain.stdout.decode("utf-8"), "café – 43 ✓ 東京\n")

    def test_an_internal_fault_is_one_json_object_not_a_traceback(self):
        (self.dir / "jev.json").write_text("{broken")
        done = self.run_cli("--json", "--check-regex", "43", "-")
        result = json.loads(done.stdout)
        self.assertEqual((done.returncode, result["outcome"], result["ok"], done.stderr), (1, "internal_error", False, ""))
        self.assertNotIn("Traceback", done.stdout)

    def test_it_runs_without_a_jev_key_or_with_jev_switched_off(self):
        env = {key: value for key, value in self.env.items() if key not in ("PIVOT_FAKE_JEV", "TYPESAFE_API_KEY")}
        done = subprocess.run(self.argv("--json", "--check-regex", "43", "-"), input="q", capture_output=True,
                              text=True, env=env, timeout=60)
        result = json.loads(done.stdout)
        self.assertEqual((done.returncode, result["slot"], done.stderr), (0, "strong@high", ""))
        self.assertIn("no Jev client", result["decision"]["fallback"])
        self.jev({"error": "must not be called"})
        off = json.loads(self.run_cli("--json", "--no-jev", "--check-regex", "43", "--difficulty", "1", "-").stdout)
        self.assertEqual((off["ok"], off["slot"], off["decision"]["route"], off["jev"]),
                         (True, "weak@low", "derived", None))
        blind = json.loads(self.run_cli("--json", "--no-jev", "--check-regex", "43", "-").stdout)
        self.assertEqual(blind["slot"], "strong@high")
        self.assertIn("--no-jev", blind["decision"]["fallback"])

    def test_doctor_and_interop_report_as_json(self):
        doctor = subprocess.run([sys.executable, "-m", "pivotlab.cli", "doctor", "--policy",
                                 str(self.dir / "policy.json")], capture_output=True, text=True, env=self.env,
                                timeout=60)
        report = json.loads(doctor.stdout)
        self.assertEqual((report["schema"], report["policy"]["reference"], report["state"]["writable"]),
                         ("jev-pivot.doctor/v1", "strong@high", True))
        sick = subprocess.run([sys.executable, "-m", "pivotlab.cli", "doctor", "--policy",
                               str(self.dir / "nope.json")], capture_output=True, text=True, env=self.env, timeout=60)
        self.assertEqual((sick.returncode, json.loads(sick.stdout)["ok"]), (1, False))
        launcher = [sys.executable, str(ROOT / "pivot")]                 # the launcher itself, from elsewhere
        bare = {key: value for key, value in self.env.items() if key != "PYTHONPATH"}
        interop = subprocess.run(launcher + ["interop"], capture_output=True, text=True, env=bare, timeout=60,
                                 cwd=self.tmp.name)
        card = json.loads(interop.stdout)
        via = subprocess.run(launcher + ["run", "--policy", str(self.dir / "policy.json"), "--providers", "fake",
                                         "-"], input="q", capture_output=True, text=True, env=bare, timeout=60,
                             cwd=self.tmp.name)
        self.assertEqual((via.returncode, via.stdout, via.stderr), (0, "ANSWER: 43\n", ""))
        self.assertEqual(card["dsco"][:4], ["dsco", "interop", "run", "generic"])
        self.assertEqual((card["dsco"][-1], Path(card["direct"][0]).name), ("-", "pivot"))


if __name__ == "__main__":
    unittest.main()
