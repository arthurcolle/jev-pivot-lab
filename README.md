# jev-pivot-lab

Answer a prompt on the cheapest model and reasoning effort that has a measured record good enough for it.

Two things live here:

- **`pivot run`**, a sub-executor an agent can spawn: prompt in, answer out. It picks a slot (a model at a
  reasoning effort, through a provider) from a policy file fitted on real measurements, falls back when a
  provider or the router is down, and always ends inside its deadline with a result that says what happened.
- **The lab** that produces the policy: it runs every model and effort your Claude Code and Codex logins
  (and optionally OpenRouter or Ramp Router) offer on exact-answer problems, replays routing policies against
  the results, and screens newly released models.

The router is [TypeSafe Jev](https://docs.typesafe.ai), a decision model that answers bounded questions about
a state. The idea of using one to pivot between weak and strong models comes from
[jev-opus](https://github.com/WXK-AI/jev-opus); this repo measures whether that pays.

## What the measurements say

43 slots (every current-generation model at every effort) on 34 exact-answer problems, tools off, one sample
per cell, 14 November 2026. Each cell is problems right, then the cost of all 34 at list price.

| Model | low | medium | high | xhigh | max | ultra |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 32 · $0.029 | 33 · $0.033 | **34 · $0.037** | 34 · $0.049 | 34 · $0.120 | |
| Claude Sonnet 5.5 | 31 · $0.37 | 32 · $0.39 | 32 · $0.43 | 34 · $0.59 | 34 · $1.62 | |
| Claude Opus 5.5 | 34 · $0.53 | 34 · $0.61 | 34 · $0.76 | 34 · $1.09 | 34 · $2.27 | |
| Claude Fable 5.2 | 28 · $1.00 | 29 · $1.03 | 34 · $1.30 | 34 · $1.82 | 34 · $3.57 | |
| GPT-6 Luna | 9 · $0.017 | 13 · $0.018 | 24 · $0.030 | 30 · $0.038 | 29 · $0.041 | |
| GPT-6.1 Terra | 32 · $0.20 | 33 · $0.22 | 34 · $0.21 | 34 · $0.24 | 34 · $0.29 | 34 · $0.23 |
| GPT-6.1 Sol | 34 · $0.43 | 34 · $0.44 | 34 · $0.48 | 34 · $0.56 | 34 · $0.60 | 34 · $0.58 |
| GPT-6 Astra | 34 · $2.14 | 34 · $2.14 | 34 · $2.26 | 34 · $2.59 | 34 · $2.86 | 34 · $2.69 |

- **Effort substitutes for model size on these problems.** Haiku 5.5 at high effort matched the largest
  models on every problem for 3% of the cost of Fable 5.2 at high.
- **Per-turn routing did not beat pinning that slot.** With an answer check to catch failures, Jev's picks
  kept 34/34 at 1.31x the cost. Without a check they were cheaper (0.91x) and got 2 of 34 wrong. A cheap slot
  that never fails leaves a router nothing to rescue.
- **Mapping difficulty to effort inside one model cost more.** A reimplementation of jev-opus's
  difficulty thresholds cost 1.26x on Haiku 5.5 and 1.34x on Fable 5.2 for the same answers.
- **Older generations lose twice.** On the one hard problem they reached, Opus 4.6, 4.7 and 4.8 used 10k to
  21k tokens where Opus 5.5 used 3k to 6k, at a higher price per token.
- **New models are mostly not better.** Of 21 models OpenRouter listed in the previous 30 days, one held up:
  `meta/muse-ember` at low effort got 34/34 for $0.024. Five were 4x to 37x the cost for the same answers,
  and the rest were less accurate, broken at the provider, or returned nothing inside the output ceiling.

These are closed-form reasoning problems (arithmetic, traces, small searches). They say nothing about
tool use, coding in a repository, or writing quality, and one sample per cell is thin.

## Use it as a sub-executor

Needs Python 3.10+ on macOS or Linux and at least one of the `claude` or `codex` CLIs, logged in. Nothing
to install.

```sh
git clone https://github.com/arthurcolle/jev-pivot-lab && cd jev-pivot-lab
./pivot doctor                      # can this machine serve requests? (JSON; add --probe for a live call)
./pivot run "What is 17 + 26?"      # prints the answer
echo "..." | ./pivot run --json -   # one JSON object: text, slot, attempts, cost, decision
```

From another agent, spawn `pivot run --json -` with the prompt on stdin. For
[dsco](https://github.com/arthurcolle/dsco):

```sh
dsco interop run generic --prompt "<text>" --stdin --json -- /path/to/jev-pivot-lab/pivot run --json -
```

What a caller can rely on:

- **Output.** Only the result goes to stdout and nothing to stderr (unless `--verbose`), so a caller that
  captures both as one stream still gets clean output.
- **Exit codes.** 0 answered, 1 run error (the JSON `outcome` says which), 2 usage, 78 configuration,
  143 terminated.
- **Deadline.** `--max-seconds` (default 300) bounds the whole run, every attempt included.
- **Termination.** On SIGTERM it stops its child processes and exits at once; children stay in its process
  group so a supervisor's group kill reaches them.
- **Degrading instead of failing.** If Jev is unreachable it uses the reference slot. A slot that errors,
  times out or returns nothing is replaced by the reference slot, then by a reliable slot on another
  provider, and is put on a cooldown that other `pivot` processes on the machine respect.
- **Money.** Subscription CLIs only by default. Metered gateways need `--allow-billed` and are capped per
  run (`--max-usd-billed`, default $0.05) and per day across all processes (`--daily-usd-billed`, $1.00).

Useful options: `--expect VALUE`, `--check-regex RE` or `--check-cmd CMD` verify the answer and let a
cheaper slot be tried first; `--models a,b` sets a floor on which models may be used; `--budget-usd`
caps the run; `--slot` pins one slot; `--no-jev` never sends the prompt to Jev; `--dry-run` prints the
plan and calls nothing. `./pivot run --help` lists everything.

It answers from reasoning alone: slots run with tools off and no file access.

## Measure your own tasks

The shipped policy comes from reasoning puzzles. To find the cheapest slot for the work you actually have,
write your tasks down with their answers, one JSON object per line:

```json
{"id": "json-path", "prompt": "In this JSON, what is the value of user.plan? ...", "expect": "team"}
{"id": "sql-count", "prompt": "Write a SQL query that returns the number of rows in orders.", "regex": "(?is)select\\s+count\\(\\s*\\*\\s*\\)\\s+from\\s+orders"}
```

`expect` is the exact final answer (numbers compare as numbers, anything else as written); `regex` passes
when the pattern is found in the reply. Then:

```sh
./pivot grid --name mine --items-file examples/tasks.jsonl --models claude:claude-haiku-5-5,codex:gpt-6-luna --efforts low,high
./pivot fit --from mine --out policy/mine.json     # needs at least 10 tasks
./pivot run --policy policy/mine.json "..."
```

On the 14 tasks in `examples/tasks.jsonl` that took 56 calls and about a cent at list prices:

| Slot | Right |
|---|---|
| `claude-haiku-5-5@low` | 14/14 |
| `claude-haiku-5-5@high` | 13/14 |
| `gpt-6-luna@low` | 13/14 |
| `gpt-6-luna@high` | 14/14 |

Haiku at high effort lost its one task by replying with the whole sorted list instead of the single value
asked for: with `expect`, the format counts. Add `--system-file` to measure under your own instructions;
`pivot run` then answers under the same ones. More tasks give a policy worth trusting; 14 is a demonstration.

## How a slot is chosen

`pivot fit` compiles a grid run into `policy/default.json`: for each of five difficulty levels, the slots
worth offering with their measured records, plus the reference slot (the cheapest one that was right on
every problem).

- **Without a check**, a wrong answer would go unnoticed, so only a slot with a clean record is used.
  With the shipped policy that is `claude-haiku-5-5@high` at every difficulty, and Jev is not called.
- **With a check**, Jev rates the request's difficulty, then picks from that level's options knowing the
  budget, the time left and the conversation. A failed check retries on the reference slot.

With a check the prompt is sent to TypeSafe as well as to the chosen model's provider. Set
`TYPESAFE_API_KEY` to enable it; without the key the reference slot is used.

## The lab

```sh
./pivot slots --models all          # every model and effort both CLIs offer on this login
./pivot grid --name full --models current --concurrency 12    # the built-in problems on every current slot
./pivot replay --name full          # score routing policies against the grid (Jev calls only)
./pivot scout --name new --against full --new-since 30 --max-usd-openrouter 3
./pivot report --name league        # reprint tables from saved evidence, at no cost
```

- Every paid call is cached in `runs/cache/`, so reruns and replays are free.
- Child processes never see `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`: a call can only be served by the CLI's
  own login. Subscription usage is reported at API list prices for scale; none of it is billed.
- `scout` screens gateway models released in a window: each runs the same problems as the measured slots
  and stops as soon as one of them beats it outright. Gateway calls are billed, with a hard cap per run.
- Only efforts a model really has become slots. Both CLIs and both gateways run an unsupported effort
  at a neighbouring level without saying so.

## Evidence in this repo

`runs/league` (43 slots x 34 problems), `runs/combined` (the first 11-slot grid with live conversation runs),
`runs/gamut9` (older generations on the first problems), `runs/scout-2026-11` (the new-model screen),
`runs/example` (the 14 example tasks), and `runs/cache` (every call's receipt).
`./pivot report --name league` reprints the table above.

## Limits

- The shipped policy is fitted on reasoning puzzles. For other work, fit one on your own tasks, set a floor
  with `--models`, or pass a check.
- Prices and model lists are a snapshot (`prices.json`, `catalog.json`); refit when they move.
  `pivot doctor` warns when the policy is over 30 days old.
- The Ramp Router adapter is written from its documentation and has never been run against the service.
- Gateway calls carry a 20,000-token output ceiling; a model that spends it all on reasoning returns nothing
  and is graded wrong.
- Wall times through the CLIs are dominated by startup (about 4 s for Claude Code, 15 s for Codex).

## Tests

```sh
python3 -m unittest discover -s tests    # 86 tests, offline: no model call, no network
```

They include the real command run as a child process against a scripted provider: output purity, exit
codes, SIGTERM cleanup, deadlines, and cooldowns shared between processes.

## License

No license has been chosen yet, so default copyright applies.
