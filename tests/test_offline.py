"""Offline tests: no CLI call, no Jev call, no network."""
import itertools
import json
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pivotlab import cli, gateways, grid as grid_mod, loop, meta, router, scout, slots, tasks  # noqa: E402

CLAUDE_OUT = json.dumps({
    "is_error": False, "result": "ANSWER: 3563525602", "total_cost_usd": 0.026088,
    "usage": {"input_tokens": 2, "cache_creation_input_tokens": 1027, "cache_read_input_tokens": 0,
              "output_tokens": 241, "output_tokens_details": {"thinking_tokens": 228}}})
CODEX_OUT = "\n".join(json.dumps(event) for event in [
    {"type": "thread.started", "thread_id": "x"},
    {"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": "ignoring a setting"}},
    {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "ANSWER: 3563525602"}},
    {"type": "turn.completed", "usage": {"input_tokens": 11094, "cached_input_tokens": 2432,
                                         "output_tokens": 150, "reasoning_output_tokens": 137}}])
A, B, C = (slots.Slot("claude", "weak", "low", 0), slots.Slot("claude", "weak", "high", 1),
           slots.Slot("codex", "strong", "max", 0))
SOLVES = {A.id: 1, B.id: 3, C.id: 4}       # highest tier each fake slot gets right
PRICE = {A.id: 0.01, B.id: 0.05, C.id: 0.40}


def find_item(message, opening):
    """Follow-up wording repeats across threads; the opening message (which holds the data) disambiguates."""
    for item in SUITE + LIVE + HOLD:
        if item["message"] == message and (item["context"][0]["user"] if item["context"] else message) == opening:
            return item
    raise LookupError(message)


def fake_runner(calls):
    def run(slot, prompt, **_):
        calls.append((slot.id, prompt))
        message = prompt.split("New user message:\n")[-1]
        opening = prompt.split("\nAssistant: ")[0].removeprefix("Conversation so far:\nUser: ")
        item = find_item(message, opening)
        text = f"ANSWER: {item['gold']}" if item["tier"] <= SOLVES[slot.id] else "ANSWER: 999999999"
        return {"slot": slot.id, "provider": slot.provider, "model": slot.model, "effort": slot.effort,
                "ok": True, "text": text, "error": None, "wall_s": 1.0, "tool_events": 0,
                "usage": {"input": 100, "cached_input": 0, "output": 10 * (item["tier"] + 1), "reasoning": 0},
                "usd_reported": PRICE[slot.id]}
    return run


class FakeJev:
    """Rates difficulty as the generator's tier, so routing logic is tested apart from Jev's judgment."""
    model = "fake-jev"

    def __init__(self):
        self.requests, self.states = 0, []

    def ask(self, state, questions):
        self.requests += 1
        opening = state["conversation"][0]["user"] if state["conversation"] else state["message"]
        item = find_item(state["message"], opening)
        answers = {}
        for qid, question in questions.items():
            if qid == "difficulty":
                answers[qid] = {"type": "score", "score": float(item["tier"])}
            elif qid == "pick":
                answers[qid] = {"type": "choice", "choice": sorted(question["criteria"])[-1]}
            elif qid == "select":
                # No cap: the reference. Any cap: the cheapest option. Shows the budget reaches Jev.
                self.states.append(state)
                capped = state["budget"]["allowance"] != "no cap"
                answers[qid] = {"type": "choice", "choice": "option0" if capped else next(
                    key for key, text in question["criteria"].items() if "(the reference)" in text)}
            else:
                answers[qid] = {"type": "noul", "noul": 0.0 if qid == "lookup" else 0.6}
        return {"answers": answers, "usage": {"input_tokens": 50}, "latency_ms": 2.0, "model": self.model}


SUITE = tasks.build_suite("unit", (2, 2, 2, 2, 2), threads=1)
LIVE = tasks.thread_items("live-unit", 0) + tasks.thread_items("live-unit", 1)
HOLD = tasks.build_suite("hold", (2, 2, 2, 2, 2), threads=0)      # same ids as SUITE, different problems


def make_grid(tmp, calls=None, **budget):  # budget: claude=, codex=, max_calls=
    cache = grid_mod.Cache(Path(tmp) / "calls.jsonl")
    caps = grid_mod.Budget({"claude": budget.get("claude", 99.0), "codex": budget.get("codex", 99.0)},
                           budget.get("max_calls", 999))
    rows = grid_mod.run_grid(SUITE, [A, B, C], cache=cache, budget=caps, prices={}, concurrency=1,
                             runner=fake_runner(calls if calls is not None else []), log=lambda _: None)
    return rows, cache, caps


class TaskTests(unittest.TestCase):
    def test_gold_values_match_brute_force(self):
        r = random.Random(3)
        for _ in range(20):
            xs = r.sample(range(3, 30), 8)
            target = sum(r.sample(xs, 3))
            brute = sum(1 for n in range(len(xs) + 1) for combo in itertools.combinations(xs, n)
                        if sum(combo) == target)
            self.assertEqual(tasks.subset_count(xs, target), brute)
        self.assertEqual(tasks.inversions([3, 1, 2]), 2)
        self.assertEqual(tasks.shortest([("A", "B", 5), ("B", "C", 5), ("A", "C", 11)], "A", "C"), 10)

    def test_every_family_is_deterministic_and_self_consistent(self):
        for tier, families in tasks.FAMILIES.items():
            for name, make in families:
                first, again = make(random.Random(name)), make(random.Random(name))
                self.assertEqual(first, again, name)
                self.assertTrue(tasks.check(f"ANSWER: {first[1]}", str(first[1])), name)
        self.assertEqual(tasks.build_suite("s"), tasks.build_suite("s"))
        self.assertNotEqual(tasks.build_suite("s")[4]["gold"], tasks.build_suite("t")[4]["gold"])
        grown = tasks.build_suite("s", extras=[("h", "lcg", 2), ("h", "program80", 1)])
        self.assertEqual(grown[:len(tasks.build_suite("s"))], tasks.build_suite("s"))
        self.assertEqual([i["id"] for i in grown[-3:]], ["t2-lcg-h0", "t2-lcg-h1", "t4-program80-h0"])
        with self.assertRaises(ValueError):
            tasks.build_suite("s", extras=[("h", "nope", 1)])
        for name, make in tasks.HARD:                     # longer families: extras only, still exact
            self.assertEqual(make(random.Random(name)), make(random.Random(name)), name)
            self.assertNotIn(name, [family for families in tasks.FAMILIES.values() for family, _ in families])
        hard = tasks.build_suite("s", (0, 0, 0, 0, 0), 0, [("p", "mul12", 1)])
        self.assertEqual((len(hard), hard[0]["tier"], hard[0]["id"]), (1, 4, "t4-mul12-p0"))
        a, b = (int(n) for n in re.findall(r"\d{12}", hard[0]["message"]))
        self.assertEqual(hard[0]["gold"], str(a * b))

    def test_answer_parsing(self):
        self.assertTrue(tasks.check("work...\nANSWER: 3,563,525,602", "3563525602"))
        self.assertTrue(tasks.check("ANSWER: 410426 (base 7)", "410426"))
        self.assertTrue(tasks.check("ANSWER: −12", "-12"))
        self.assertTrue(tasks.check("41", "41"))
        self.assertFalse(tasks.check("ANSWER: 42", "41"))
        self.assertFalse(tasks.check("", "41"))

    def test_thread_context_carries_gold_or_actual_replies(self):
        gold = tasks.thread_items("s", 0)
        self.assertEqual(gold[0]["context"], [])
        self.assertIn(f"ANSWER: {gold[0]['gold']}", gold[1]["prompt"])
        heavy = tasks.thread_items("s", 0, kind="heavy")
        self.assertEqual([i["tier"] for i in heavy], [4, 0, 1, 4, 0, 2])
        self.assertEqual((heavy[0]["id"], heavy[1]["gold"]), ("heavy-0-turn1", "28"))
        self.assertEqual(heavy, tasks.thread_items("s", 0, kind="heavy"))
        live = tasks.thread_items("s", 0, ["ANSWER: wrong"])
        self.assertIn("ANSWER: wrong", live[1]["prompt"])
        self.assertTrue(live[1]["prompt"].endswith(live[1]["message"]))


class SlotTests(unittest.TestCase):
    def test_parsers_read_real_cli_shapes(self):
        claude = slots.parse_claude(CLAUDE_OUT, "", 0)
        self.assertEqual((claude["ok"], claude["usage"]["output"], claude["usage"]["reasoning"]), (True, 241, 228))
        self.assertEqual(claude["usage"]["input"], 1029)
        codex = slots.parse_codex(CODEX_OUT, "", 0)
        self.assertEqual((codex["ok"], codex["text"], codex["tool_events"]), (True, "ANSWER: 3563525602", 0))
        self.assertEqual(codex["usage"], {"input": 11094, "cached_input": 2432, "output": 150, "reasoning": 137})
        self.assertFalse(slots.parse_codex('{"type":"turn.failed","error":{"message":"limit"}}', "", 1)["ok"])
        self.assertFalse(slots.parse_claude("not json", "boom", 1)["ok"])

    def test_tool_use_is_counted(self):
        events = CODEX_OUT + "\n" + json.dumps({"type": "item.completed", "item": {"type": "command_execution"}})
        self.assertEqual(slots.parse_codex(events, "", 0)["tool_events"], 1)

    def test_usd_uses_reported_cost_then_list_prices(self):
        prices = {"strong": {"input": 10.0, "cached_input": 1.0, "output": 50.0}}
        codex = {"model": "strong", "usd_reported": None,
                 "usage": {"input": 11094, "cached_input": 2432, "output": 150, "reasoning": 137}}
        self.assertAlmostEqual(slots.usd(codex, prices) or 0.0, (8662 * 10 + 2432 * 1 + 150 * 50) / 1e6)
        self.assertEqual(slots.usd({"usd_reported": 0.02}, prices), 0.02)
        self.assertAlmostEqual(slots.cost(codex, "usd_flat", prices), (11094 * 10 + 150 * 50) / 1e6)
        self.assertIsNone(slots.usd({"model": "unknown", "usd_reported": None, "usage": {}}, prices))
        with self.assertRaises(ValueError):
            slots.cost({"slot": "x", "model": "unknown", "usd_reported": None, "usage": {}}, "usd", prices)

    def test_discovery_splits_every_effort_into_a_slot(self):
        with mock.patch.object(slots, "claude_efforts", return_value=["low", "max"]), \
                mock.patch.object(slots, "codex_efforts", return_value=["low", "ultra"]):
            found = slots.discover()
            self.assertEqual([s.id for s in found], ["claude-fable-5-2@low", "claude-fable-5-2@max",
                                                     "gpt-6-astra@low", "gpt-6-astra@ultra"])
            self.assertEqual([s.id for s in slots.discover(efforts=["ultra"])], ["gpt-6-astra@ultra"])
        with self.assertRaises(ValueError):
            slots.discover(("bogus",))

    def test_all_expands_to_every_catalogued_model_with_only_real_efforts(self):
        with mock.patch.object(slots, "claude_efforts", return_value=["low", "high", "xhigh", "max"]), \
                mock.patch.object(slots, "claude_catalog", return_value={"old": [], "mid": ["low", "high", "max"]}), \
                mock.patch.object(slots, "codex_catalog", return_value={"tiny": ["low", "high"]}):
            self.assertEqual([s.id for s in slots.discover(("all",))],
                             ["old@default", "mid@low", "mid@high", "mid@max", "tiny@low", "tiny@high"])
            self.assertEqual([s.id for s in slots.discover(("codex:all",))], ["tiny@low", "tiny@high"])
            self.assertEqual([s.id for s in slots.discover(("claude:unlisted",))][0], "unlisted@low")
            self.assertEqual(len(slots.discover(("all", "claude:mid"))), 6)          # no duplicate slots
            with mock.patch.object(slots, "current_models", return_value=[("claude", "mid"), ("codex", "tiny")]):
                self.assertEqual([s.id for s in slots.discover(("current",))],
                                 ["mid@low", "mid@high", "mid@max", "tiny@low", "tiny@high"])
        self.assertEqual([model for _, model in slots.current_models()][:1], ["claude-haiku-5-5"])
        with tempfile.TemporaryDirectory() as scratch:
            plain = slots.command_for(slots.Slot("claude", "old", slots.NO_EFFORT, 0), "s", Path(scratch))
            self.assertNotIn("--effort", plain)
            self.assertIn("--effort", slots.command_for(A, "s", Path(scratch)))

    def test_an_answer_from_another_model_is_not_filed_under_the_slot(self):
        def served_by(usage):
            out = json.dumps({**json.loads(CLAUDE_OUT), "modelUsage": usage})
            with tempfile.TemporaryDirectory() as scratch, mock.patch.object(slots, "SCRATCH", Path(scratch)), \
                    mock.patch.object(slots, "_spawn", return_value=(out, "", 0)):
                return slots.run_slot(A, "q")

        swapped = served_by({"other-model": {"canonicalModel": "other-model"}})
        self.assertFalse(swapped["ok"])
        self.assertIn("served by other-model", swapped["error"])
        self.assertTrue(served_by({"weak-20260101": {"canonicalModel": "weak"}})["ok"])
        self.assertTrue(served_by({})["ok"])

    def test_child_never_sees_metered_keys_and_tools_are_off(self):
        seen = {}

        def fake_run(command, prompt, **kwargs):
            seen[command[0]] = (command, kwargs["env"])
            return (CLAUDE_OUT if command[0] == "claude" else CODEX_OUT), "", 0

        with tempfile.TemporaryDirectory() as scratch, \
                mock.patch.object(slots, "SCRATCH", Path(scratch)), \
                mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k1", "OPENAI_API_KEY": "k2",
                                               "CLAUDE_CODE_EFFORT_LEVEL": "max"}), \
                mock.patch.object(slots, "_spawn", side_effect=fake_run):
            self.assertTrue(slots.run_slot(A, "q")["ok"])
            self.assertTrue(slots.run_slot(C, "q")["ok"])
        for command, env in seen.values():
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("CLAUDE_CODE_EFFORT_LEVEL", env)       # it would outrank --effort
        claude, codex = seen["claude"][0], seen["codex"][0]
        self.assertEqual(claude[claude.index("--tools") + 1], "")
        self.assertIn('model_reasoning_effort="max"', codex)
        self.assertIn("shell_tool", codex)
        self.assertEqual(codex[-1], "-")


class GridTests(unittest.TestCase):
    def test_cache_means_a_cell_is_paid_for_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            rows, _, caps = make_grid(tmp, calls)
            self.assertEqual(len(calls), len(SUITE) * 3)
            self.assertEqual(caps.summary()["calls"], len(calls))
            again, _, caps2 = make_grid(tmp, calls)
            self.assertEqual(len(calls), len(SUITE) * 3)
            self.assertEqual(caps2.summary()["calls"], 0)
            self.assertTrue(all(row["cached"] for row in again))
            self.assertEqual([r["correct"] for r in rows], [r["correct"] for r in again])

    def test_budget_caps_skip_instead_of_spending(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _, caps = make_grid(tmp, codex=0.5)
            skipped = [row for row in rows if row["status"] == "skipped"]
            self.assertTrue(skipped and all(row["slot"] == C.id for row in skipped))
            self.assertLessEqual(caps.summary()["usd_equivalent"]["codex"], 0.5 + PRICE[C.id])
        with tempfile.TemporaryDirectory() as tmp:
            rows, _, caps = make_grid(tmp, max_calls=5)
            self.assertEqual(sum(row["status"] != "skipped" for row in rows), 5)

    def test_breaker_stops_an_unserved_slot_and_a_rate_limited_provider(self):
        gone, calls = slots.Slot("claude", "gone", "low", 0), []

        def runner(slot, prompt, **kwargs):
            calls.append(slot.id)
            if slot.id == gone.id:
                return {"slot": slot.id, "provider": slot.provider, "model": slot.model, "effort": slot.effort,
                        "ok": False, "text": "", "error": "model not available", "wall_s": 0.1, "tool_events": 0,
                        "usage": {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}, "usd_reported": None}
            return fake_runner([])(slot, prompt)

        with tempfile.TemporaryDirectory() as tmp:
            breaker = grid_mod.Breaker()
            rows = grid_mod.run_grid(SUITE, [A, gone], cache=grid_mod.Cache(Path(tmp) / "c.jsonl"),
                                     budget=grid_mod.Budget({"claude": 99.0}, 999), prices={}, concurrency=1,
                                     runner=runner, breaker=breaker, log=lambda _: None)
            self.assertEqual(calls.count(gone.id), 2)
            self.assertEqual(sum(row["status"] == "skipped" for row in rows), len(SUITE) - 2)
            self.assertEqual(list(breaker.summary()["stopped_slots"]), [gone.id])
            self.assertEqual(breaker.summary()["stopped_providers"], [])      # a dead model is not a rate limit
            grid = router.Grid(SUITE, rows, "usd", {})
            self.assertEqual((grid.dead_slots, grid.slot_ids, len(grid.items)), ([gone.id], [A.id], len(SUITE)))
        limited = grid_mod.Breaker(provider_limit=3)
        failure = {"ok": False, "error": "429"}
        for receipt in ({"ok": True}, failure, failure):
            limited.record(A, receipt)
        limited.record(B, {"ok": True, "error": None})
        self.assertIsNone(limited.blocked(B))                                  # a success resets the run of errors
        for slot in (A, B, B):
            limited.record(slot, failure)
        self.assertIn("claude stopped", limited.blocked(B) or "")
        self.assertIsNone(limited.blocked(C))
        self.assertEqual(limited.summary()["stopped_providers"], ["claude"])

    def test_partial_grids_still_span_tiers(self):
        order = grid_mod.execution_order(SUITE)
        self.assertEqual(sorted({item["tier"] for item in order[:5]}), [0, 1, 2, 3, 4])

    def test_timeouts_are_cached_until_a_longer_limit_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = grid_mod.Cache(Path(tmp) / "c.jsonl")
            cache.put("k", {"ok": False, "error": "timeout", "timeout_s": 60})
            self.assertIsNotNone(grid_mod.Cache(Path(tmp) / "c.jsonl").get("k", 60))
            self.assertIsNone(cache.get("k", 300))


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rows, _, _ = make_grid(self.tmp.name)
        self.grid = router.Grid(SUITE, rows, "usd", {})
        self.fake = FakeJev()
        self.jev = router.CachedJev(self.fake, Path(self.tmp.name) / "jev.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ladder_keeps_only_slots_that_buy_accuracy(self):
        stats = router.slot_stats(self.grid, self.grid.items)
        self.assertEqual(router.build_ladder(stats), [A.id, B.id, C.id])
        stats[B.id]["accuracy"] = stats[A.id]["accuracy"]
        self.assertEqual(router.build_ladder(stats), [A.id, C.id])

    def test_bucket_map_is_monotone_and_conservative_on_unseen_difficulty(self):
        ladder = [A.id, B.id, C.id]
        tiers = {item["id"]: item["tier"] for item in self.grid.items}
        self.assertEqual(router.bucket_map(self.grid, ladder, self.grid.items, tiers), {0: 0, 1: 0, 2: 1, 3: 1, 4: 2})
        easy = [item for item in self.grid.items if item["tier"] <= 1]
        self.assertEqual(router.bucket_map(self.grid, ladder, easy, tiers), {0: 0, 1: 0, 2: 2, 3: 2, 4: 2})

    def test_hysteresis_matches_the_jev_opus_rule(self):
        self.assertEqual(router.apply_hysteresis(0, 2, 0), (2, 1))       # raise at once, then hold
        self.assertEqual(router.apply_hysteresis(2, 0, 1), (2, 0))       # held for one more turn
        self.assertEqual(router.apply_hysteresis(2, 0, 0), (1, 0))       # step down one rung at a time
        self.assertEqual(router.apply_hysteresis(1, 1, 1), (1, 0))

    def test_replay_scores_policies_against_the_grid(self):
        result = router.replay(self.grid, self.jev, log=lambda _: None)
        p, n = result["policies"], len(SUITE)
        self.assertEqual(p[f"always:{C.id}"]["correct"], n)
        self.assertLess(p[f"always:{A.id}"]["correct"], n)
        self.assertEqual(p["oracle"]["correct"], n)
        self.assertEqual(p["jev-difficulty"]["correct"], n)
        self.assertLess(p["jev-difficulty"]["cost"], p[f"always:{C.id}"]["cost"])
        self.assertAlmostEqual(p["jev-difficulty"]["cost"], p["oracle"]["cost"])
        self.assertEqual(p["cascade"]["correct"], n)
        self.assertGreater(p["cascade"]["attempts"], n)
        self.assertGreaterEqual(p["jev-difficulty+esc+hyst"]["cost"], p["jev-difficulty+esc"]["cost"])
        self.assertEqual(p["jev-choice"]["correct"], n)                  # fake always picks the top rung
        self.assertEqual(result["rank_correlation"]["jev_difficulty_vs_slots_wrong"],
                         result["rank_correlation"]["tier_label_vs_slots_wrong"])

    def test_calibration_never_sees_the_item_being_routed(self):
        plans = router.plan_items(self.grid, self.jev, log=lambda _: None)
        hardest = [item for item in self.grid.items if item["tier"] == 4]
        lone = router.Grid([i for i in SUITE if i["tier"] < 4] + hardest[:1],
                           [r for r in self.grid.cells.values()
                            if r["item"] in {i["id"] for i in SUITE if i["tier"] < 4} | {hardest[0]["id"]}],
                           "usd", {})
        plan = router.plan_items(lone, self.jev, log=lambda _: None)[hardest[0]["id"]]
        self.assertNotIn(C.id, plan["ladder"])      # without its own outcome, nothing shows C is needed
        self.assertIn(C.id, plans[hardest[0]["id"]]["ladder"])

    def test_a_second_run_can_be_a_held_out_test_of_a_fit_made_elsewhere(self):
        self.assertEqual(HOLD[0]["id"], SUITE[0]["id"])
        self.assertNotEqual(HOLD[0]["prompt"], SUITE[0]["prompt"])
        rows = grid_mod.run_grid(HOLD, [A, B, C], cache=grid_mod.Cache(Path(self.tmp.name) / "hold.jsonl"),
                                 budget=grid_mod.Budget({"claude": 99.0, "codex": 99.0}, 999), prices={},
                                 concurrency=1, runner=fake_runner([]), log=lambda _: None)
        held = router.Grid(HOLD, rows, "usd", {})
        plans = router.plan_items(held, self.jev, log=lambda _: None, calibration=self.grid)
        whole = router.slot_stats(self.grid, self.grid.items)
        for plan in plans.values():                      # one fit, from every calibration problem
            self.assertEqual(plan["ladder"], router.build_ladder(whole))
            self.assertEqual(plan["reference"], C.id)
        hardest = next(item for item in HOLD if item["tier"] == 4)
        self.assertEqual(plans[hardest["id"]]["bucket"], 4)                 # its own difficulty, not its namesake's
        result = router.replay(held, self.jev, log=lambda _: None, calibration=self.grid)
        p, n = result["policies"], len(HOLD)
        self.assertIn("held out", result["fit"])
        self.assertEqual((p["jev-bucket-cheapest"]["correct"], p["derived+check"]["correct"]), (n, n))
        self.assertLess(p["jev-bucket-cheapest"]["cost"], p[f"always:{C.id}"]["cost"])
        self.assertEqual(router.replay(self.grid, self.jev, log=lambda _: None)["fit"], "leave-one-out")

    def test_jev_answers_are_cached_on_disk(self):
        router.replay(self.grid, self.jev, log=lambda _: None)
        first = self.fake.requests
        again = router.CachedJev(self.fake, Path(self.tmp.name) / "jev.jsonl")
        router.replay(self.grid, again, log=lambda _: None)
        self.assertEqual(self.fake.requests, first)
        self.assertEqual(again.fresh["requests"], 0)

    def test_incomplete_items_are_dropped_and_timeouts_charged(self):
        rows = [dict(row) for row in self.grid.cells.values()]
        rows[0].update(status="skipped", correct=False)
        victim = next(r for r in rows if r["slot"] == C.id and r["item"] != rows[0]["item"])
        victim.update(status="timeout", correct=False, ok=False)
        grid = router.Grid(SUITE, rows, "usd", {})
        self.assertEqual(grid.dropped, [rows[0]["item"]])
        self.assertEqual(grid.imputed, 1)
        self.assertEqual(grid.cost(victim["item"], C.id), PRICE[C.id])

    def test_bucket_best_follows_cost_shape_not_one_ladder(self):
        # X: tiny overhead, heavy reasoning. Y: big overhead, light reasoning. Z: cheapest, wrong when hard.
        items = [{"id": f"i{n}", "tier": tier, "family": "f", "thread": None} for n, tier in enumerate([0, 0, 4, 4])]
        price = {"X": {0: 0.01, 4: 0.50}, "Y": {0: 0.05, 4: 0.20}, "Z": {0: 0.005, 4: 0.01}}
        rows = [{"item": item["id"], "slot": slot, "status": "ok", "wall_s": 1.0, "usd_reported": price[slot][item["tier"]],
                 "correct": slot != "Z" or item["tier"] == 0} for item in items for slot in price]
        grid = router.Grid(items, rows, "usd", {})
        fitted = router.bucket_best(grid, router.slot_stats(grid, items), items, {i["id"]: i["tier"] for i in items})
        self.assertEqual(fitted["reliable"], ["Y", "X"])
        self.assertEqual((fitted["best"][0], fitted["best"][4]), ("X", "Y"))
        self.assertNotIn("Z", fitted["best"].values())

    def test_unpriced_and_filtered_slots_are_left_out(self):
        rows = [dict(row) for row in self.grid.cells.values()]
        for row in rows:
            if row["slot"] == C.id:
                row["usd_reported"] = None
        weak_only = {"weak": {"input": 1.0, "cached_input": 0.1, "output": 5.0}}
        grid = router.Grid(SUITE, rows, "usd_flat", weak_only)
        self.assertEqual((grid.unpriced_slots, grid.slot_ids), ([C.id], [A.id, B.id]))
        self.assertEqual(len(grid.items), len(SUITE))
        self.assertEqual(router.Grid(SUITE, rows, "out_tokens", {}, only=["weak@*"]).slot_ids, [A.id, B.id])
        self.assertEqual(router.Grid(SUITE, rows, "out_tokens", {}).slot_ids, [A.id, B.id, C.id])

    def test_bucket_cheapest_hands_easy_buckets_to_weak_slots(self):
        items, tiers = self.grid.items, {item["id"]: item["tier"] for item in self.grid.items}
        reliable = router.bucket_best(self.grid, router.slot_stats(self.grid, items), items, tiers)["best"]
        self.assertEqual(set(reliable.values()), {C.id})
        cheap = router.bucket_cheapest(self.grid, items, tiers, reliable)
        self.assertEqual(cheap, {0: A.id, 1: A.id, 2: B.id, 3: B.id, 4: C.id})
        easy = [item for item in items if item["tier"] == 0]
        thin = [item for item in items if item["tier"] != 0] + easy[:router.MIN_SUPPORT - 1]
        self.assertEqual(router.bucket_cheapest(self.grid, thin, tiers, reliable)[0], C.id)   # too few to trust A
        costs = {A.id: 1.0, B.id: 2.0, C.id: 9.0, "off": 1.5}
        self.assertEqual(router.climb([A.id, C.id], costs, A.id), [C.id])
        self.assertEqual(router.climb([A.id, B.id, C.id], costs, "off"), [B.id, C.id])
        p, n = router.replay(self.grid, self.jev, log=lambda _: None)["policies"], len(SUITE)
        self.assertEqual((p["jev-bucket-cheapest"]["correct"], p["jev-bucket-cheapest+esc"]["correct"]), (n, n))
        self.assertLess(p["jev-bucket-cheapest"]["cost"], p[f"always:{C.id}"]["cost"])

    def test_opus_rule_uses_the_published_thresholds(self):
        self.assertEqual([router.opus_rule_effort(d) for d in (0.5, 1.25, 2.24, 2.25, 3.09, 3.1, 4.0)],
                         ["low", "medium", "medium", "high", "high", "xhigh", "xhigh"])

    def test_spearman(self):
        self.assertAlmostEqual(router.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(router.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(router.spearman([1, 1, 1], [1, 2, 3]))


class LoopTests(unittest.TestCase):
    def test_live_conversation_pivots_escalates_and_carries_real_replies(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cache, _ = make_grid(tmp)
            grid = router.Grid(SUITE, rows, "usd", {})
            jev = router.CachedJev(FakeJev(), Path(tmp) / "jev.jsonl")
            fitted = loop.fit(grid, jev)
            fitted["map_jev"][4] = 1                    # force an under-estimate on the hardest turn
            calls, by_id = [], {s.id: s for s in (A, B, C)}
            budget = grid_mod.Budget({"claude": 99.0, "codex": 99.0}, 999)
            run = loop.live_thread("live-unit", 0, fitted, by_id, jev, cache=cache, budget=budget, prices={},
                                   route="ladder", runner=fake_runner(calls), log=lambda _: None)
            tried = {t["family"]: t["tried"] for t in run["turns"]}
            self.assertEqual(run["totals"]["correct"], run["totals"]["n"])
            self.assertEqual(tried["thread-subsets10"], [B.id, C.id])         # escalated after a failed check
            self.assertEqual(tried["thread-length"], [A.id])                  # cheap follow-up went cheap
            self.assertEqual(tried["thread-max"], [C.id])                     # held one turn after the raise
            self.assertEqual(tried["thread-oddsum"], [B.id])                  # then stepped down one rung
            fixed = loop.live_thread("live-unit", 0, fitted, by_id, jev, cache=cache, budget=budget, prices={},
                                     fixed=C.id, runner=fake_runner(calls), log=lambda _: None)
            self.assertEqual(fixed["totals"]["attempts"], fixed["totals"]["n"])
            self.assertGreater(fixed["totals"]["usd"], run["totals"]["usd"])
            bucket = loop.live_thread("live-unit", 0, fitted, by_id, jev, cache=cache, budget=budget, prices={},
                                      runner=fake_runner(calls), log=lambda _: None)
            self.assertEqual({t["tried"][0] for t in bucket["turns"]}, {C.id})   # only C is reliable on this grid
            self.assertEqual(bucket["totals"]["correct"], bucket["totals"]["n"])
            cheap = loop.live_thread("live-unit", 0, fitted, by_id, jev, cache=cache, budget=budget, prices={},
                                     route="cheapest", runner=fake_runner(calls), log=lambda _: None)
            first = {t["family"]: t["tried"][0] for t in cheap["turns"]}
            self.assertEqual((first["thread-length"], first["thread-subsets10"]), (A.id, C.id))
            self.assertEqual(cheap["totals"]["correct"], cheap["totals"]["n"])
            self.assertLess(cheap["totals"]["usd"], bucket["totals"]["usd"])
            weak = loop.live_thread("live-unit", 0, fitted, by_id, jev, cache=cache, budget=budget, prices={},
                                    fixed=A.id, runner=fake_runner(calls), log=lambda _: None)
            self.assertFalse(weak["turns"][0]["correct"])
            self.assertIn("Assistant: ANSWER: 999999999", calls[-1][1])        # its wrong reply stays in the transcript


OPENROUTER_LIST = {"data": [
    {"id": "acme/thinker", "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000002"},
     "supported_parameters": ["reasoning", "max_tokens"],
     "reasoning": {"supported_efforts": ["high", "low", "medium"], "default_effort": "medium"}},
    {"id": "acme/plain", "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}, "supported_parameters": []},
    {"id": "acme/vague", "pricing": {"prompt": "0.000001", "completion": "0.000002"},
     "supported_parameters": ["reasoning"], "reasoning": None},
    {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}, "supported_parameters": ["reasoning"]}]}
RAMP_LIST = {"object": "list", "data": [
    {"id": "claude-haiku-5-5", "router": {"surfaces": ["responses", "messages"],
     "pricing": {"input": "0.10", "output": "0.50", "cache_read_input": "0.01"},
     "capabilities": {"reasoning": {"supported": True, "efforts": [{"value": "max"}, {"value": "none"},
                                                                    {"value": "low"}]}}}},
    {"id": "jev-1.14.0", "router": {"surfaces": ["systemone"], "pricing": {"input": "0.04", "output": "0"}}},
    {"id": "no-price", "router": {"surfaces": ["responses"], "pricing": {}}}]}
OPENROUTER_REPLY = {"model": "acme/thinker", "provider": "Acme", "choices": [{"message": {"content": "ANSWER: 43"}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 90, "cost": 0.00114,
                              "prompt_tokens_details": {"cached_tokens": 20},
                              "completion_tokens_details": {"reasoning_tokens": 80}}}
RAMP_REPLY = {"model": "claude-haiku-5-5", "output": [
    {"type": "reasoning", "summary": []},
    {"type": "message", "content": [{"type": "output_text", "text": "ANSWER: 43"}]}],
    "usage": {"input_tokens": 110, "output_tokens": 70, "input_tokens_details": {"cached_tokens": 0},
              "output_tokens_details": {"reasoning_tokens": 60}}}


class GatewayTests(unittest.TestCase):
    def test_gateway_catalogs_give_real_efforts_and_prices_per_million(self):
        listed = gateways.parse_openrouter_models(OPENROUTER_LIST)
        self.assertEqual(listed["acme/thinker"]["efforts"], ["low", "medium", "high"])
        self.assertEqual(listed["acme/thinker"]["price"], {"input": 2.0, "output": 10.0, "cached_input": 0.2})
        self.assertEqual(listed["acme/plain"]["efforts"], [])
        self.assertEqual(listed["acme/vague"]["efforts"], ["low", "medium", "high"])
        self.assertNotIn("openrouter/auto", listed)                      # a router alias, not a priced model
        ramp = gateways.parse_ramp_models(RAMP_LIST)
        self.assertEqual(list(ramp), ["claude-haiku-5-5"])               # priced, and served on /responses
        self.assertEqual(ramp["claude-haiku-5-5"]["efforts"], ["none", "low", "max"])
        self.assertEqual(ramp["claude-haiku-5-5"]["price"]["cached_input"], 0.01)

    def test_gateway_slots_are_named_for_their_gateway_and_carry_a_price(self):
        both = {"openrouter": gateways.parse_openrouter_models(OPENROUTER_LIST),
                "ramp": gateways.parse_ramp_models(RAMP_LIST)}
        with mock.patch.object(gateways, "catalog", side_effect=lambda provider, scratch: both[provider]):
            found = slots.discover(("openrouter:acme/thinker", "openrouter:acme/plain", "ramp:claude-haiku-5-5"))
            self.assertEqual([s.id for s in found],
                             ["openrouter:acme/thinker@low", "openrouter:acme/thinker@medium",
                              "openrouter:acme/thinker@high", "openrouter:acme/plain@default",
                              "ramp:claude-haiku-5-5@none", "ramp:claude-haiku-5-5@low", "ramp:claude-haiku-5-5@max"])
            self.assertTrue(all(s.metered and s.price for s in found))
            self.assertEqual(found[0].price_table(), {"input": 2.0, "cached_input": 0.2, "output": 10.0})
            with self.assertRaises(ValueError):
                slots.discover(("openrouter:acme/unlisted",))
            with self.assertRaises(ValueError):
                slots.discover(("openrouter:all",))                      # 470 billed models is never a default
        self.assertFalse(A.metered)
        self.assertEqual(slots.worst_case_usd(A, "s", "p"), 0.0)
        self.assertAlmostEqual(slots.worst_case_usd(found[0], "s" * 300, "p" * 300),
                               (200 * 2.0 + gateways.MAX_OUTPUT_TOKENS * 10.0) / 1e6)

    def test_requests_and_replies_follow_each_gateway_s_api(self):
        chat = gateways.request_body("openrouter", "acme/thinker", "high", "sys", "q")
        self.assertEqual((chat["reasoning"], chat["max_tokens"], chat["messages"][0]["role"]),
                         ({"effort": "high"}, gateways.MAX_OUTPUT_TOKENS, "system"))
        self.assertNotIn("reasoning", gateways.request_body("openrouter", "acme/plain", "default", "sys", "q"))
        responses = gateways.request_body("ramp", "claude-haiku-5-5", "low", "sys", "q")
        self.assertEqual((responses["instructions"], responses["input"], responses["reasoning"]),
                         ("sys", "q", {"effort": "low"}))
        got = gateways.parse_reply("openrouter", 200, OPENROUTER_REPLY, {})
        self.assertEqual((got["ok"], got["text"], got["usd_reported"], got["served"]),
                         (True, "ANSWER: 43", 0.00114, ["acme/thinker"]))
        self.assertEqual(got["usage"], {"input": 120, "cached_input": 20, "output": 90, "reasoning": 80})
        ramp = gateways.parse_reply("ramp", 200, RAMP_REPLY, {"X-Router-Provider": "anthropic"})
        self.assertEqual((ramp["ok"], ramp["text"], ramp["upstream"], ramp["usage"]["reasoning"]),
                         (True, "ANSWER: 43", "anthropic", 60))
        refused = gateways.parse_reply("ramp", 402, {"error": {"message": "no available credit"}}, {})
        self.assertEqual((refused["ok"], refused["error"]), (False, "HTTP 402: no available credit"))
        self.assertFalse(gateways.parse_reply("openrouter", 200, {"choices": [{"message": {"content": ""}}]}, {})["ok"])
        spent = gateways.parse_reply("openrouter", 200, {"choices": [{"message": {"content": None}}], "usage": {
            "prompt_tokens": 90, "completion_tokens": gateways.MAX_OUTPUT_TOKENS}}, {})
        self.assertEqual((spent["ok"], spent["text"], spent["truncated"]), (True, "", True))   # billed, graded wrong
        self.assertFalse(tasks.check(spent["text"], "41"))
        self.assertFalse(got["truncated"])

    def test_a_metered_call_uses_the_gateway_key_and_records_the_price_it_paid(self):
        slot = slots.Slot("openrouter", "acme/thinker", "high", 2, (2.0, 0.2, 10.0))
        sent = {}

        def post(url, key, body, timeout):
            sent.update(url=url, key=key, body=body)
            return 200, OPENROUTER_REPLY, {}

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key"}), \
                mock.patch.object(gateways, "http_json", post), \
                mock.patch.object(gateways.call, "__defaults__", (post,)):
            receipt = slots.run_slot(slot, "q")
        self.assertEqual((sent["url"], sent["key"]), ("https://openrouter.ai/api/v1/chat/completions", "or-key"))
        self.assertEqual((receipt["ok"], receipt["slot"], receipt["price"]["output"]),
                         (True, "openrouter:acme/thinker@high", 10.0))
        self.assertEqual(slots.usd(receipt, {}), 0.00114)                               # the gateway's own figure
        rounded = {**receipt, "usd_reported": 0.0}                                      # seen live: cost came back 0.0
        self.assertAlmostEqual(slots.usd(rounded, {}), (100 * 2.0 + 20 * 0.2 + 90 * 10.0) / 1e6)
        self.assertAlmostEqual(slots.cost(receipt, "usd_flat", {}), (120 * 2.0 + 90 * 10.0) / 1e6)
        with mock.patch.dict("os.environ", {}, clear=True):
            missing = slots.run_slot(slots.Slot("ramp", "claude-haiku-5-5", "low", 0, (0.1, 0.01, 0.5)), "q")
        self.assertEqual((missing["ok"], missing["error"]), (False, "RAMP_ROUTER_API_KEY is not set"))

    def test_billed_spend_cannot_pass_its_cap_even_with_calls_in_flight(self):
        budget = grid_mod.Budget({"openrouter": 1.0, "claude": 1.0}, 99)
        self.assertTrue(budget.allow("openrouter", reserve=0.4))
        self.assertTrue(budget.allow("openrouter", reserve=0.4))
        self.assertFalse(budget.allow("openrouter", reserve=0.4))        # 1.2 could be owed: not sent
        budget.add("openrouter", 0.05, reserve=0.4)                      # one settled far under its worst case
        self.assertTrue(budget.allow("openrouter", reserve=0.4))
        self.assertFalse(budget.allow("openrouter", reserve=0.4))
        self.assertTrue(budget.allow("claude"))                          # subscription calls are not reserved
        self.assertEqual(budget.summary()["usd_equivalent"]["openrouter"], 0.05)
        slot = slots.Slot("openrouter", "acme/thinker", "high", 2, (2.0, 0.2, 10.0))
        tight = grid_mod.Budget({"openrouter": 0.05}, 99)                # under one call's worst case
        with tempfile.TemporaryDirectory() as tmp:
            row = grid_mod.run_cell(SUITE[0], slot, cache=grid_mod.Cache(Path(tmp) / "c.jsonl"), budget=tight,
                                    prices={}, timeout=5, runner=lambda *a, **k: self.fail("must not be sent"))
        self.assertEqual((row["status"], row["reason"]), ("skipped", "budget cap"))


class ScoutTests(unittest.TestCase):
    """New slots against A (cheap, weak), B (mid) and C (dear, right on everything)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rows, self.cache, _ = make_grid(self.tmp.name)
        self.grid = router.Grid(SUITE, rows, "usd", {})
        self.order = scout.order_items(self.grid)

    def tearDown(self):
        self.tmp.cleanup()

    def new(self, name, effort, rank, solves, price):
        slot = slots.Slot("openrouter", name, effort, rank, (1.0, 0.1, 1.0))
        SOLVES[slot.id], PRICE[slot.id] = solves, price
        return slot

    def run_scout(self, new_slots, calls=None, **kwargs):
        budget = grid_mod.Budget({"openrouter": 99.0}, 9999)
        return scout.scout(new_slots, self.grid, metric="usd", prices={}, cache=self.cache, budget=budget,
                           breaker=grid_mod.Breaker(), timeout=5, concurrency=1,
                           runner=fake_runner(calls if calls is not None else []), log=lambda _: None, **kwargs)

    def test_cheap_problems_come_first_then_the_most_discriminating(self):
        warm, rest = self.order[:scout.WARMUP], self.order[scout.WARMUP:]
        self.assertTrue(all(item["tier"] <= 1 for item in warm))               # every incumbent solves these
        self.assertEqual(rest[0]["tier"], 4)                                   # only C solves these
        self.assertEqual(sorted(i["id"] for i in self.order), sorted(i["id"] for i in SUITE))

    def test_verdicts_good_expensive_crap_and_broken(self):
        calls = []
        good = self.new("new/good", "high", 0, 4, 0.02)             # right on everything, 20x cheaper than C
        pricey = self.new("new/pricey", "high", 0, 4, 5.0)          # right on everything, 12x the price of C
        weak = self.new("new/weak", "high", 0, 0, 0.30)             # tier 0 only, dearer than B
        got = {row["slot"]: row for row in self.run_scout([good, pricey, weak], calls)}
        self.assertEqual(got[good.id]["verdict"], "good")
        self.assertEqual((got[good.id]["ran"], got[good.id]["right"]), (len(SUITE), len(SUITE)))
        self.assertEqual(got[good.id]["rival"], C.id)
        self.assertEqual(got[pricey.id]["verdict"], "expensive")
        self.assertEqual(got[pricey.id]["ran"], scout.CHECKPOINTS[0])           # caught on the cheap warm-up
        self.assertEqual(got[weak.id]["verdict"], "crap")
        self.assertLess(got[weak.id]["ran"], len(SUITE))
        self.assertEqual(len([c for c in calls if c[0] == pricey.id]), scout.CHECKPOINTS[0])

        def down(slot, prompt, **kwargs):
            return {"slot": slot.id, "provider": slot.provider, "model": slot.model, "effort": slot.effort,
                    "ok": False, "text": "", "error": "HTTP 404: no such model", "wall_s": 0.1, "tool_events": 0,
                    "usage": {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}, "usd_reported": None}

        dead = slots.Slot("openrouter", "new/dead", "high", 0, (1.0, 0.1, 1.0))
        budget = grid_mod.Budget({"openrouter": 99.0}, 99)
        row = scout.scout([dead], self.grid, metric="usd", prices={}, cache=self.cache, budget=budget,
                          breaker=grid_mod.Breaker(), timeout=5, concurrency=1, runner=down, log=lambda _: None)[0]
        self.assertEqual((row["verdict"], row["ran"]), ("broken", 2))

    def test_a_finished_slot_that_only_matches_a_cheaper_one_is_beaten(self):
        same = self.new("new/same", "high", 0, 4, 0.45)             # as right as C, a little dearer than C
        row = self.run_scout([same])[0]
        self.assertEqual((row["verdict"], row["rival"], row["ran"]), ("beaten", C.id, len(SUITE)))
        cheap_weak = self.new("new/tiny", "high", 0, 1, 0.001)      # weaker than C but cheaper than everything
        self.assertEqual(self.run_scout([cheap_weak])[0]["verdict"], "tradeoff")   # a new cheap point, not a win
        near = self.new("new/near", "high", 0, 3, 0.45)             # misses only tier 4, costs more than C
        row = self.run_scout([near])[0]
        self.assertEqual((row["verdict"], row["rival"]), ("beaten", C.id))      # C is more accurate for less
        self.assertGreaterEqual(row["right"] / row["ran"], scout.CRAP_BELOW)

    def test_efforts_are_bracketed_not_all_run(self):
        low, mid, high, top = (self.new("new/m", effort, rank, solves, price) for rank, (effort, solves, price)
                               in enumerate([("low", 4, 0.01), ("medium", 4, 0.02), ("high", 4, 0.03),
                                             ("max", 4, 0.09)]))
        self.assertEqual([s.effort for s in scout.bracket([low, mid, high, top])], ["high", "low", "medium", "max"])
        got = {row["effort"]: row for row in self.run_scout([low, mid, high, top])}
        self.assertEqual((got["high"]["verdict"], got["low"]["verdict"]), ("good", "good"))
        self.assertEqual(got["max"]["verdict"], "not run")                      # high already answers everything
        self.assertIn("already answers everything", got["max"]["note"])
        self.assertEqual(got["medium"]["verdict"], "not run")                   # so does the cheaper low
        bad = [self.new("new/bad", effort, rank, 0, 0.30) for rank, effort in enumerate(["low", "high", "max"])]
        calls = []
        rows = self.run_scout(bad, calls)
        self.assertEqual([row["verdict"] for row in rows], ["not run", "crap", "not run"])
        self.assertEqual({call[0] for call in calls}, {bad[1].id})              # only `high` cost anything

    def test_new_models_are_recent_untested_and_not_variants(self):
        now = __import__("time").time()
        listed = {"a/fresh": {"created": now - 86400}, "a/fresh:free": {"created": now - 86400},
                  "a/old": {"created": now - 90 * 86400}, "openai/fresh": {"created": now - 86400},
                  "a/tested": {"created": now - 86400}, "a/video": {"created": now - 86400, "text": False},
                  "~a/alias": {"created": now - 86400}, "b/newer": {"created": now - 3600}}
        found = scout.new_models("openrouter", listed, 30, ("anthropic", "openai"), {("openrouter", "a/tested")})
        self.assertEqual(found, ["b/newer", "a/fresh"])

    def test_table_names_the_incumbent_each_slot_is_measured_against(self):
        good = self.new("new/good2", "high", 0, 4, 0.02)
        text = scout.table(self.run_scout([good]) + [{"slot": "x@max", "verdict": "not run", "note": "why"}], "usd")
        self.assertIn(f"x the cost of {C.id}", text)
        self.assertIn("why", text)


class BudgetLoopTests(unittest.TestCase):
    def test_select_route_tells_jev_the_running_budget_and_the_previous_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cache, _ = make_grid(tmp)
            grid = router.Grid(SUITE, rows, "usd", {})
            fake = FakeJev()
            jev = router.CachedJev(fake, Path(tmp) / "jev.jsonl")
            fitted = loop.fit(grid, jev)
            self.assertEqual(fitted["reference"], C.id)
            self.assertAlmostEqual(fitted["reference_cost"], PRICE[C.id])      # sums differ in the last bit by version
            common = {"cache": cache, "budget": grid_mod.Budget({"claude": 99.0, "codex": 99.0}, 999),
                      "prices": {}, "runner": fake_runner([]), "log": lambda _: None}
            by_id = {s.id: s for s in (A, B, C)}
            free = loop.live_thread("live-unit", 0, fitted, by_id, jev, route="select", **common)
            self.assertEqual({t["tried"][0] for t in free["turns"]}, {C.id})      # no cap: the fake takes the reference
            capped = loop.live_thread("live-unit", 0, fitted, by_id, jev, route="select", budget_x=0.5, **common)
            turns = capped["totals"]["n"]
            self.assertEqual({t["tried"][0] for t in capped["turns"]}, {A.id})    # capped: the fake takes the cheapest
            self.assertEqual(capped["totals"]["correct"], turns)                  # the check sent failures to C
            self.assertTrue(all(t["tried"][-1] == C.id for t in capped["turns"] if len(t["tried"]) > 1))
            told = [state for state in fake.states if state["budget"]["allowance"] != "no cap"]
            self.assertEqual(len(told), turns)
            self.assertAlmostEqual(told[0]["budget"]["total"], 0.5 * PRICE[C.id] * turns)
            self.assertEqual(told[0]["budget"]["spent"], 0.0)
            self.assertAlmostEqual(told[1]["budget"]["spent"], capped["turns"][0]["cost"])
            self.assertEqual(told[1]["budget"]["turns_left_after_this"], turns - 2)
            self.assertEqual(told[1]["context"]["previous_slot"], capped["turns"][0]["tried"][-1])
            self.assertTrue(told[1]["context"]["previous_check_passed"])
            self.assertNotIn("previous_slot", told[0]["context"])
            derived = loop.live_thread("live-unit", 0, fitted, by_id, jev, route="derived", **common)
            self.assertEqual(derived["totals"]["correct"], turns)
            self.assertLess(derived["totals"]["usd"], free["totals"]["usd"])
            hurried = loop.live_thread("live-unit", 1, fitted, by_id, jev, route="select", latency_target=5.0,
                                       **{**common, "runner": fake_runner([])})
            self.assertEqual(hurried["totals"]["n"], turns)
            self.assertEqual(fake.states[-1]["budget"]["latency_target_s"], 5.0)

    def test_allowance_follows_what_the_reference_would_need(self):
        fitted = {"unit": "usd", "reference_cost": 1.0}
        heavy, light = {"cost": 4.0}, {"cost": 0.1}
        kwargs = {"budget_x": 0.5, "turns": 4, "latency_target": None}
        _, first = loop.live_budget(fitted, heavy, turn=0, spent=0.0, **kwargs)
        self.assertAlmostEqual(first, 2.0 * 4.0 / (4.0 + 3.0))                    # a heavy turn gets a heavy share
        _, small = loop.live_budget(fitted, light, turn=0, spent=0.0, **kwargs)
        self.assertLess(small, first)
        _, last = loop.live_budget(fitted, light, turn=3, spent=0.5, **kwargs)
        self.assertAlmostEqual(last, 1.5)                                         # savings carry to the last turn
        state, none = loop.live_budget(fitted, light, budget_x=None, turns=4, turn=1, spent=0.3, latency_target=9.0)
        self.assertEqual((none, state["allowance"], state["latency_target_s"]), (None, "no cap", 9.0))


class MetaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rows, _, _ = make_grid(self.tmp.name)
        self.grid = router.Grid(SUITE, rows, "usd", {})
        self.tiers = {item["id"]: item["tier"] for item in SUITE}
        self.card = meta.cards(self.grid, SUITE, self.tiers)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cards_record_each_slot_per_bucket_and_shrink_thin_evidence(self):
        weak = self.card[A.id]
        self.assertEqual((weak["buckets"][0]["solved"], weak["buckets"][0]["n"]), (4, 4))
        self.assertEqual(weak["buckets"][4]["solved"], 0)
        self.assertLess(weak["buckets"][0]["p"], 1.0)                 # perfect on easy, but weak overall
        self.assertEqual(self.card[C.id]["buckets"][0]["p"], 1.0)
        self.assertAlmostEqual(weak["buckets"][0]["cost"], PRICE[A.id])

    def test_options_are_the_cheapest_first_frontier_and_always_hold_the_reference(self):
        easy = meta.options(self.card, 0, C.id)
        self.assertEqual([o["slot"] for o in easy], [A.id, B.id, C.id])
        hard = meta.options(self.card, 4, C.id)
        self.assertEqual(hard[-1]["slot"], C.id)
        self.assertEqual(meta.options(self.card, 0, A.id)[0]["slot"], A.id)
        fast = meta.options(self.card, 0, C.id, time_value=1.0)       # every fake call takes one second
        self.assertEqual([o["slot"] for o in fast], [A.id, B.id, C.id])
        crowd = {f"s{n}": {"solved": n, "n": 40, "overall": n / 40,
                           "buckets": {0: {"n": 5, "solved": 5, "p": 0.5 + n / 80, "cost": 1.0 + n, "wall": 1.0}}}
                 for n in range(30)}
        thinned = meta.options(crowd, 0, "s29")
        self.assertEqual(len(thinned), meta.MAX_OPTIONS)
        self.assertEqual((thinned[0]["slot"], thinned[-1]["slot"]), ("s0", "s29"))

    def test_derive_trades_cost_against_measured_reliability(self):
        opts = [{"slot": "weak", "p": 0.90, "cost": 1.0, "price": 1.0},
                {"slot": "mid", "p": 0.97, "cost": 4.0, "price": 4.0},
                {"slot": "ref", "p": 1.00, "cost": 10.0, "price": 10.0}]
        self.assertEqual(meta.derive(opts, "ref"), "mid")                              # within the risk margin
        self.assertEqual(meta.derive(opts, "ref", risk=0.0), "ref")
        self.assertEqual(meta.derive(opts, "ref", checked=True), "weak")               # 1 + 0.1 * 10 beats 4.3
        self.assertEqual(meta.derive(opts, "ref", allowance=2.0), "weak")              # all the budget affords
        self.assertEqual(meta.derive(opts, "ref", allowance=0.5), "weak")              # nothing fits: cheapest
        self.assertEqual(meta.derive(opts, "ref", allowance=20.0), "mid")

    def test_jev_sees_the_budget_the_context_and_the_cards(self):
        fake, item = FakeJev(), next(i for i in SUITE if i["context"])
        jev = router.CachedJev(fake, Path(self.tmp.name) / "jev.jsonl")
        opts = meta.options(self.card, 0, C.id)
        common = {"context": meta.turn_context(item, previous_slot=A.id), "checked": False}
        capped = meta.ask(jev, item, 0, opts, C.id, budget=meta.static_budget(opts[-1], "usd", 0.25), **common)
        free = meta.ask(jev, item, 0, opts, C.id, budget=meta.static_budget(opts[-1], "usd", None), **common)
        self.assertEqual((capped, free), (A.id, C.id))
        state = fake.states[0]
        self.assertAlmostEqual(state["budget"]["allowance"], 0.25 * PRICE[C.id])
        self.assertEqual(state["budget"]["reference_cost"], PRICE[C.id])
        self.assertEqual((state["context"]["prior_turns"], state["context"]["previous_slot"]),
                         (len(item["context"]), A.id))
        self.assertGreater(state["context"]["context_tokens_estimate"], 0)
        self.assertEqual(meta.ask(jev, item, 0, opts[-1:], C.id, budget={}, context={}, checked=True), C.id)

    def test_replay_scores_derived_and_jev_selection_at_each_budget(self):
        jev = router.CachedJev(FakeJev(), Path(self.tmp.name) / "jev.jsonl")
        result = router.replay(self.grid, jev, log=lambda _: None)
        p, n = result["policies"], len(SUITE)
        for name in ("derived", "derived+check", "derived@0.5x", "derived@0.25x",
                     "jev-select", "jev-select+check", "jev-select@0.5x", "jev-select@0.25x"):
            self.assertIn(name, p)
        self.assertEqual(p["jev-select"]["correct"], n)                                 # the fake picks the reference
        self.assertAlmostEqual(p["jev-select"]["cost"], p[f"always:{C.id}"]["cost"])
        self.assertLess(p["jev-select@0.25x"]["cost"], p["jev-select"]["cost"])         # the cap reached Jev
        self.assertLess(p["jev-select@0.25x"]["correct"], n)                            # and cost it accuracy
        self.assertEqual(p["derived+check"]["correct"], n)                              # a check restores it
        self.assertLess(p["derived+check"]["cost"], p[f"always:{C.id}"]["cost"])
        self.assertGreaterEqual(p["derived+check"]["attempts"], n)
        self.assertEqual(result["selection"][0]["reference"], C.id)


class ReportTests(unittest.TestCase):
    def test_many_slots_print_as_a_model_by_effort_matrix_and_a_trimmed_policy_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _, _ = make_grid(tmp)
            text = cli.matrix_tables(SUITE, rows, "usd", {})
            header, weak, strong = (text.splitlines()[n] for n in (1, 2, 3))
            self.assertEqual(header.split(), ["model", "low", "high", "max"])
            self.assertEqual(weak.split()[0], "weak")
            self.assertEqual(strong.split(), ["strong", "-", "-", f"{len(SUITE)}/{len(SUITE)}"])
            grid = router.Grid(SUITE, rows, "usd", {})
            result = router.replay(grid, router.CachedJev(FakeJev(), Path(tmp) / "jev.jsonl"), log=lambda _: None)
            for n in range(9):                       # nine dominated fixed slots that should not fill the table
                result["policies"][f"always:pad{n}@low"] = {**result["policies"][f"always:{A.id}"], "cost": 50.0 + n}
            short, full = cli.replay_table(result), cli.replay_table(result, everything=True)
            self.assertIn("always:pad8@low", full)
            self.assertNotIn("always:pad8@low", short)
            self.assertIn("more always:<...> rows hidden", short)
            self.assertIn(f"always:{C.id}", short)   # the best fixed slot always shows


if __name__ == "__main__":
    unittest.main()
