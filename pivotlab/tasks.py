"""Seeded problems with exact gold answers, standalone and as multi-turn conversations.

Every gold value is computed here in code, so a slot is graded by a deterministic check and
never by another model. Tier is the generator's prior about difficulty (0 trivial .. 4 extreme);
the grid measures the real difficulty.
"""
from __future__ import annotations

import heapq
import json
import random
import re
from pathlib import Path

ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
INT_RE = re.compile(r"-?\d[\d,_]*")
PRIMES = [503, 617, 733, 829, 941, 997]
RULE110 = {(1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 1)}


def inversions(xs: list[int]) -> int:
    return sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[i] > xs[j])


def subset_count(xs: list[int], target: int) -> int:
    ways = [1] + [0] * target
    for x in xs:
        for total in range(target, x - 1, -1):
            ways[total] += ways[total - x]
    return ways[target]


def shortest(edges: list[tuple[str, str, int]], source: str, goal: str) -> int:
    graph: dict[str, list[tuple[str, int]]] = {}
    for a, b, w in edges:
        graph.setdefault(a, []).append((b, w))
        graph.setdefault(b, []).append((a, w))
    best, queue = {source: 0}, [(0, source)]
    while queue:
        dist, node = heapq.heappop(queue)
        if node == goal:
            return dist
        if dist > best.get(node, dist):
            continue
        for nxt, w in graph.get(node, []):
            if dist + w < best.get(nxt, dist + w + 1):
                best[nxt] = dist + w
                heapq.heappush(queue, (dist + w, nxt))
    raise ValueError("graph is not connected")


def add2(r):
    a, b = r.randint(11, 99), r.randint(11, 99)
    return f"What is {a} + {b}?", a + b


def maxlist(r):
    xs = r.sample(range(10, 100), 5)
    return f"What is the largest number in this list: {xs}?", max(xs)


def mul(digits):
    def make(r):
        a, b = (r.randint(10 ** (digits - 1), 10 ** digits - 1) for _ in range(2))
        return f"Compute {a} * {b} exactly.", a * b
    return make


def mul32(r):
    a, b = r.randint(101, 999), r.randint(11, 99)
    return f"Compute {a} * {b} exactly.", a * b


def sumlist(r):
    xs = [r.randint(10, 99) for _ in range(8)]
    return f"What is the sum of these numbers: {xs}?", sum(xs)


def lcg(r):
    m, a, b, k = r.choice([97, 101, 103, 107, 109, 113]), r.randint(3, 9), r.randint(1, 20), 10
    x = start = r.randint(1, m - 1)
    for _ in range(k):
        x = (a * x + b) % m
    return f"Let x = {start}. Repeat {k} times: x = ({a}*x + {b}) mod {m}. What is the final x?", x


def base7(r):
    n = value = r.randint(10000, 99999)
    digits = ""
    while n:
        digits, n = str(n % 7) + digits, n // 7
    return f"Write the decimal number {value} in base 7. Give only the base-7 digits.", digits


def count_inversions(size):
    def make(r):
        xs = r.sample(range(10, 100), size)
        return ("Count the inversions in this list, meaning pairs of positions i < j whose values "
                f"satisfy L[i] > L[j]: L = {xs}"), inversions(xs)
    return make


def modexp(r):
    a, b, m = r.randint(12, 98), r.randint(150, 950), r.choice(PRIMES)
    return f"Compute {a}^{b} mod {m}.", pow(a, b, m)


def path(nodes, extra):
    def make(r):
        names = [chr(ord("A") + i) for i in range(nodes)]
        order = names[1:-1]
        r.shuffle(order)
        chain = [names[0]] + order + [names[-1]]
        edges = {tuple(sorted(pair)): r.randint(6, 20) for pair in zip(chain, chain[1:])}
        while len(edges) < nodes - 1 + extra:
            pair = tuple(sorted(r.sample(names, 2)))
            edges.setdefault(pair, r.randint(1, 20))
        listed = [(a, b, w) for (a, b), w in edges.items()]
        r.shuffle(listed)
        text = ", ".join(f"{a}-{b}: {w}" for a, b, w in listed)
        question = (f"An undirected weighted graph has these edges (node-node: weight): {text}. "
                    f"What is the length of the shortest path from {names[0]} to {names[-1]}?")
        return question, shortest(listed, names[0], names[-1])
    return make


def program(steps):
    def make(r):
        x, y = start = (r.randint(2, 9), r.randint(2, 9))
        for i in range(1, steps + 1):
            if (x + i) % 3 == 0:
                x = x + 2 * y
            else:
                y, x = y + (x % 7), x + i
        return (f"Trace this program by hand.\nx = {start[0]}\ny = {start[1]}\n"
                f"for i from 1 to {steps} inclusive:\n    if (x + i) is divisible by 3:\n"
                f"        x = x + 2*y\n    else:\n        y = y + (x mod 7)\n        x = x + i\n"
                f"What is the final value of x - y?"), x - y
    return make


def subsets(size):
    def make(r):
        xs = sorted(r.sample(range(3, 40), size))
        target = sum(r.sample(xs, size // 2))
        return f"How many subsets of {xs} sum to exactly {target}?", subset_count(xs, target)
    return make


def rule110(width, steps):
    def make(r):
        cells = start = [r.randint(0, 1) for _ in range(width)]
        total = 0
        for _ in range(steps):
            cells = [1 if (cells[i - 1], cells[i], cells[(i + 1) % width]) in RULE110 else 0
                     for i in range(width)]
            total += sum(cells)
        return (f"{width} cells sit on a ring, so the first and last cells are neighbours. Each generation, "
                f"a cell becomes 1 if its (left, self, right) values are 110, 101, 011, 010 or 001, and 0 "
                f"otherwise; all cells update at once. Start: {''.join(map(str, start))}. Count the 1-cells "
                f"in each of generations 1 through {steps} and report the total of those counts."), total
    return make


def knapsack(size):
    def make(r):
        items = [(r.randint(3, 15), r.randint(10, 60)) for _ in range(size)]
        capacity = max(8, sum(w for w, _ in items) * 2 // 5)
        best = [0] * (capacity + 1)
        for w, v in items:
            for c in range(capacity, w - 1, -1):
                best[c] = max(best[c], best[c - w] + v)
        text = ", ".join(f"(weight {w}, value {v})" for w, v in items)
        return (f"Items: {text}. Each item may be used at most once and total weight may not exceed "
                f"{capacity}. What is the maximum total value?"), best[capacity]
    return make


FAMILIES = {
    0: [("add2", add2), ("maxlist", maxlist)],
    1: [("mul32", mul32), ("sumlist", sumlist)],
    2: [("lcg", lcg), ("base7", base7), ("inversions10", count_inversions(10))],
    3: [("program24", program(24)), ("modexp", modexp), ("path9", path(9, 7)),
        ("inversions18", count_inversions(18)), ("mul6", mul(6))],
    4: [("subsets14", subsets(14)), ("rule110", rule110(20, 10)), ("mul8", mul(8)),
        ("knapsack10", knapsack(10)), ("program45", program(45)),
        ("rule110x", rule110(28, 16)), ("program80", program(80))],
}
# Longer versions of the tier-4 families, reachable only through `extras`: the standard suite
# stays as it was, and these exist to find where a cheap slot at high effort stops being right.
HARD = [("mul12", mul(12)), ("subsets20", subsets(20)), ("rule110xx", rule110(40, 30)),
        ("program200", program(200)), ("knapsack16", knapsack(16)), ("inversions40", count_inversions(40)),
        ("path16", path(16, 20))]
BY_NAME = {name: (tier, make) for tier, families in FAMILIES.items() for name, make in families}
BY_NAME.update({name: (4, make) for name, make in HARD})


def render(context: list[dict], message: str) -> str:
    if not context:
        return message
    lines = ["Conversation so far:"]
    for turn in context:
        lines += [f"User: {turn['user']}", f"Assistant: {turn['assistant']}"]
    return "\n".join(lines) + f"\n\nNew user message:\n{message}"


def _item(item_id, family, tier, message, gold, context=(), thread=None, turn=None) -> dict:
    context = list(context)
    return {"id": item_id, "family": family, "tier": tier, "message": message, "gold": str(gold),
            "context": context, "prompt": render(context, message), "thread": thread, "turn": turn}


def heavy_turns(seed, index: int) -> list[dict]:
    """Two long exact computations with trivial follow-ups between them and a short one at the end."""
    r = random.Random(f"{seed}:heavy:{index}")
    ring, ring_gold = rule110(28, 16)(r)
    trace, trace_gold = program(80)(r)
    step, step_gold = lcg(r)
    start_row = re.search(r"Start: ([01]+)", ring).group(1)
    return [
        {"family": "heavy-rule110x", "tier": 4, "gold": ring_gold, "message": ring},
        {"family": "heavy-width", "tier": 0, "gold": len(start_row), "message": "How many cells are on that ring?"},
        {"family": "heavy-startcount", "tier": 1, "gold": start_row.count("1"),
         "message": "How many 1-cells were in the starting row?"},
        {"family": "heavy-program80", "tier": 4, "gold": trace_gold, "message": "Separate problem. " + trace},
        {"family": "heavy-startx", "tier": 0, "gold": re.search(r"x = (\d+)", trace).group(1),
         "message": "What value did x start with in that program?"},
        {"family": "heavy-lcg", "tier": 2, "gold": step_gold, "message": "One more. " + step},
    ]


def thread_turns(seed, index: int, kind: str = "list") -> list[dict]:
    """A conversation whose hard turns are separated by trivial follow-ups about the same data."""
    if kind == "heavy":
        return heavy_turns(seed, index)
    r = random.Random(f"{seed}:thread:{index}")
    xs = r.sample(range(10, 100), 14)
    target = sum(r.sample(xs[:10], 4))
    ordered = sorted(xs)
    return [
        {"family": "thread-inversions14", "tier": 3, "gold": inversions(xs),
         "message": ("Here is a list: L = " + str(xs) + ". Count its inversions, meaning pairs of positions "
                     "i < j whose values satisfy L[i] > L[j].")},
        {"family": "thread-length", "tier": 0, "gold": len(xs), "message": "How many numbers are in that list?"},
        {"family": "thread-sum3", "tier": 1, "gold": sum(xs[:3]),
         "message": "What is the sum of the first three numbers in the list?"},
        {"family": "thread-subsets10", "tier": 4, "gold": subset_count(xs[:10], target),
         "message": f"How many subsets of the first ten numbers of the list sum to exactly {target}?"},
        {"family": "thread-max", "tier": 0, "gold": max(xs),
         "message": "What is the largest number in the original list?"},
        {"family": "thread-oddsum", "tier": 2, "gold": sum(ordered[::2]),
         "message": ("Sort the original list in ascending order. What is the sum of the numbers in the "
                     "1st, 3rd, 5th and every other odd position of the sorted list?")},
    ]


def thread_items(seed, index: int, answers: list[str] | None = None, kind: str = "list") -> list[dict]:
    """Turn k carries turns < k as context; with no answers given, the context holds the gold replies."""
    turns, items, context = thread_turns(seed, index, kind), [], []
    name = f"{'thread' if kind == 'list' else kind}-{index}"
    for k, turn in enumerate(turns):
        items.append(_item(f"{name}-turn{k + 1}", turn["family"], turn["tier"], turn["message"],
                           turn["gold"], context, name, k))
        reply = answers[k] if answers and k < len(answers) else f"ANSWER: {turn['gold']}"
        context = context + [{"user": turn["message"], "assistant": reply}]
    return items


def build_suite(seed=7, per_tier=(1, 1, 2, 3, 2), threads: int = 1, extras=()) -> list[dict]:
    """`extras` adds (seed, family, count) problems on top, so a grid can grow without changing old items."""
    items = []
    for tier, count in enumerate(per_tier):
        families = FAMILIES[tier]
        for k in range(count):
            name, make = families[k % len(families)]
            message, gold = make(random.Random(f"{seed}:{name}:{k // len(families)}"))
            items.append(_item(f"t{tier}-{name}-{k // len(families)}", name, tier, message, gold))
    for index in range(threads):
        items += thread_items(seed, index)
    for extra_seed, name, count in extras:
        if name not in BY_NAME:
            raise ValueError(f"unknown family {name!r}; known: {sorted(BY_NAME)}")
        tier, make = BY_NAME[name]
        for k in range(int(count)):
            message, gold = make(random.Random(f"{extra_seed}:{name}:{k}"))
            items.append(_item(f"t{tier}-{name}-{extra_seed}{k}", name, tier, message, gold))
    return items


def extract(text: str) -> str:
    text = (text or "").replace("−", "-")
    lines = ANSWER_RE.findall(text) or [line for line in text.strip().splitlines() if line.strip()][-1:]
    if not lines:
        return ""
    line = re.sub(r"\([^)]*\)", "", lines[-1])
    numbers = INT_RE.findall(line)
    return numbers[-1].replace(",", "").replace("_", "") if numbers else line.strip().lower()


def check(text: str, gold: str) -> bool:
    answer = extract(text)
    try:
        return int(answer) == int(gold)
    except ValueError:
        return answer == gold.strip().lower()


def _plain(value: str) -> str:
    return value.strip().strip("`'\"").rstrip(".").strip().lower()


def matches(text: str, expect: str) -> bool:
    """Is the final answer the expected value? A whole number is compared as a number; anything
    else (a word, a date, a version) as the answer line itself, ignoring case, quotes and a full stop."""
    try:
        int(str(expect).replace(",", ""))
        return check(text, str(expect).replace(",", ""))
    except ValueError:
        lines = ANSWER_RE.findall(text or "") or [line for line in (text or "").strip().splitlines() if line.strip()][-1:]
        return bool(lines) and _plain(lines[-1]) == _plain(str(expect))


def grade(item: dict, text: str) -> bool:
    """A task with a `regex` passes when the pattern is found anywhere in the reply; any other
    task when its final answer is the expected value."""
    if item.get("regex"):
        return bool(re.search(item["regex"], text or ""))
    return matches(text, item["gold"])


def load_items(path) -> list[dict]:
    """Your own tasks, one JSON object per line: {"prompt": ..., "expect": ...} or {"prompt": ...,
    "regex": ...}; optional "id", "tier" (0 easy .. 4 hard, a guess is fine) and "context" (earlier
    turns as [{"user": ..., "assistant": ...}])."""
    items, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: not JSON ({error})") from error
        if not isinstance(task, dict) or not str(task.get("prompt", "")).strip():
            raise ValueError(f"{path}:{number}: a task needs a \"prompt\"")
        if ("expect" in task) == ("regex" in task):
            raise ValueError(f"{path}:{number}: a task needs exactly one of \"expect\" or \"regex\"")
        if "regex" in task:
            try:
                re.compile(task["regex"])
            except re.error as error:
                raise ValueError(f"{path}:{number}: bad regex ({error})") from error
        item_id = str(task.get("id") or f"task-{number}")
        if item_id in seen:
            raise ValueError(f"{path}:{number}: duplicate id {item_id!r}")
        seen.add(item_id)
        tier = int(task.get("tier", 2))
        if not 0 <= tier <= 4:
            raise ValueError(f"{path}:{number}: tier must be 0..4")
        gold = task["expect"] if "expect" in task else f"/{task['regex']}/"
        item = _item(item_id, str(task.get("family") or "custom"), tier, str(task["prompt"]), gold,
                     task.get("context") or ())
        if "regex" in task:
            item["regex"] = task["regex"]
        items.append(item)
    if not items:
        raise ValueError(f"{path}: no tasks")
    return items
