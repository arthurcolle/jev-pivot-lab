"""(model, effort) slots over the subscription CLIs: one stateless call per slot run.

`all` widens the slot set to every model each login serves: Claude models from catalog.json,
Codex models from the CLI's own model cache.

Child processes never see ANTHROPIC_API_KEY or OPENAI_API_KEY, so a call can only be served
by the CLI's own subscription login, never by a metered key.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import gateways

SYSTEM = ("Solve the problem by careful reasoning. You have no tools and must not run code. "
          "Reply with exactly one line: ANSWER: <value>")
METERED_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
# Claude Code lets this variable outrank --effort, which would make every effort slot the same slot.
OVERRIDES = ("CLAUDE_CODE_EFFORT_LEVEL",)
NO_EFFORT = "default"      # the one slot of a model that has no effort setting
DEFAULT_MODELS = ("claude:claude-fable-5-2", "codex:gpt-6-astra")
FALLBACK_EFFORTS = {"claude": ["low", "medium", "high", "xhigh", "max"],
                    "codex": ["low", "medium", "high", "xhigh", "max", "ultra"]}
# Codex features that add tools or context; off so a slot answers from reasoning alone.
CODEX_OFF = ("shell_tool", "unified_exec", "apps", "plugins", "memories", "goals", "multi_agent",
             "view_image", "image_generation", "browser_use", "in_app_browser",
             "browser_use_external", "shell_snapshot")
PRICES_PATH = Path(__file__).resolve().parents[1] / "prices.json"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalog.json"
# One fixed, empty working directory outside any repository: both CLIs put the working directory
# into the prompt, so a fresh temp dir per call would defeat provider prompt caching.
SCRATCH = Path(os.environ.get("PIVOT_SCRATCH") or Path.home() / ".cache" / "jev-pivot-lab")


@dataclass(frozen=True)
class Slot:
    provider: str
    model: str
    effort: str
    rank: int
    price: tuple | None = None      # metered gateways only: (input, cached input, output) dollars per million

    @property
    def id(self) -> str:
        # A gateway can serve the very model a subscription CLI serves: its slots carry the gateway's name.
        prefix = f"{self.provider}:" if self.provider in gateways.GATEWAYS else ""
        return f"{prefix}{self.model}@{self.effort}"

    @property
    def metered(self) -> bool:
        return self.provider in gateways.GATEWAYS

    def price_table(self) -> dict | None:
        return dict(zip(("input", "cached_input", "output"), self.price)) if self.price else None


def worst_case_usd(slot: Slot, system: str, prompt: str) -> float:
    """What one call could cost in billed money at most; nothing for the subscription CLIs."""
    table = slot.price_table()
    return gateways.worst_case_usd(table, system, prompt) if slot.metered and table else 0.0


def claude_efforts() -> list[str]:
    try:
        text = subprocess.run(["claude", "--help"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    match = re.search(r"--effort <level>.*?\(([a-z,\s]+)\)", text, re.DOTALL)
    levels = [word for word in re.split(r"[,\s]+", match.group(1)) if word] if match else []
    return levels or list(FALLBACK_EFFORTS["claude"])


def claude_catalog() -> dict[str, list[str]]:
    """Claude model -> the effort levels Claude Code documents for it ([] when it has none)."""
    try:
        models = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["claude"]["models"]
        return {str(model): [str(level) for level in levels] for model, levels in models.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def codex_catalog(hidden: bool = False) -> dict[str, list[str]]:
    """Each model the Codex CLI lists for this login -> its efforts. Models the CLI marks hidden are
    left out unless asked for: `all` should mean what the CLI itself offers."""
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        catalog = json.loads((home / "models_cache.json").read_text(encoding="utf-8"))
        return {str(m["slug"]): [str(level["effort"] if isinstance(level, dict) else level)
                                 for level in m.get("supported_reasoning_levels") or []]
                for m in catalog["models"] if hidden or m.get("visibility") != "hide"}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def codex_efforts(model: str) -> list[str]:
    return codex_catalog(hidden=True).get(model) or list(FALLBACK_EFFORTS["codex"])


def current_models() -> list[tuple[str, str]]:
    """The newest model of each line, per catalog.json."""
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return [(provider, str(model)) for provider in ("claude", "codex") for model in catalog[provider]["current"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def expand(specs) -> list[tuple[str, str]]:
    """`all`, `claude:all` and `codex:all` stand for every model the catalog or the CLI lists;
    `current` for the newest model of each line."""
    pairs = []
    for spec in specs:
        provider, _, model = spec.partition(":")
        if spec == "current":
            pairs += current_models()
        elif spec == "all":
            pairs += [("claude", name) for name in claude_catalog()] + [("codex", name) for name in codex_catalog()]
        elif provider in FALLBACK_EFFORTS and model == "all":
            pairs += [(provider, name) for name in (claude_catalog() if provider == "claude" else codex_catalog())]
        elif (provider in FALLBACK_EFFORTS or provider in gateways.GATEWAYS) and model and model != "all":
            pairs.append((provider, model))
        else:
            raise ValueError(f"model spec must be all, current, claude:<model|all>, codex:<model|all>, "
                             f"openrouter:<model> or ramp:<model>, got {spec!r}")
    return list(dict.fromkeys(pairs))


def discover(specs=DEFAULT_MODELS, efforts: list[str] | None = None) -> list[Slot]:
    """Every reasoning effort a model really has becomes its own slot."""
    slots, accepted, documented, listed = [], None, claude_catalog(), {}
    for provider, model in expand(specs):
        price = None
        if provider in gateways.GATEWAYS:
            if provider not in listed:
                listed[provider] = gateways.catalog(provider, SCRATCH)
            entry = listed[provider].get(model)
            if entry is None:
                raise ValueError(f"{provider} does not list a priced model named {model!r}")
            levels = entry["efforts"] or [NO_EFFORT]
            price = tuple(entry["price"][key] for key in ("input", "cached_input", "output"))
        elif provider == "claude":
            accepted = claude_efforts() if accepted is None else accepted
            known = documented.get(model)
            # Only levels the installed CLI accepts and the model is documented to honour.
            levels = accepted if known is None else ([e for e in accepted if e in known] or [NO_EFFORT])
        else:
            levels = codex_efforts(model)
        for rank, effort in enumerate(levels):
            if efforts is None or effort in efforts:
                slots.append(Slot(provider, model, effort, rank, price))
    if not slots:
        raise ValueError("no slots left after filtering efforts")
    return slots


def load_prices(path: Path = PRICES_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def usd(receipt: dict, prices: dict) -> float | None:
    """API-list-equivalent dollars: the CLI's own figure when it reports one, else tokens x list price."""
    reported = receipt.get("usd_reported")
    if reported is not None and not receipt.get("price"):
        return float(reported)
    price, usage = receipt.get("price") or prices.get(receipt.get("model")), receipt.get("usage") or {}
    if not price or not usage:
        return None if reported is None else float(reported)
    cached = usage.get("cached_input", 0)
    priced = ((usage.get("input", 0) - cached) * price["input"] + cached * price["cached_input"]
              + usage.get("output", 0) * price["output"]) / 1_000_000
    # A metered gateway can report a cost rounded down to zero: billed spend is never counted
    # below what its own price list says the tokens cost.
    return max(float(reported or 0.0), priced)


def cost(receipt: dict, metric: str, prices: dict) -> float:
    if metric == "usd":
        value = usd(receipt, prices)
        if value is None:
            raise ValueError(f"no usd figure for {receipt.get('slot')}; add its model to prices.json "
                             f"or use --metric out_tokens")
        return value
    if metric == "usd_flat":
        # Every input token at the uncached list price: comparisons no longer depend on which
        # call happened to hit a warm provider prompt cache.
        price, usage = receipt.get("price") or prices.get(receipt.get("model")), receipt.get("usage") or {}
        if not price:
            raise ValueError(f"no list price for {receipt.get('model')}; add it to prices.json")
        return (usage.get("input", 0) * price["input"] + usage.get("output", 0) * price["output"]) / 1_000_000
    if metric == "out_tokens":
        return float((receipt.get("usage") or {}).get("output", 0))
    if metric == "wall_s":
        return float(receipt.get("wall_s", 0.0))
    raise ValueError(f"unknown cost metric {metric!r}")


def _usage(input_: int = 0, cached: int = 0, output: int = 0, reasoning: int = 0) -> dict:
    return {"input": int(input_ or 0), "cached_input": int(cached or 0),
            "output": int(output or 0), "reasoning": int(reasoning or 0)}


def parse_claude(stdout: str, stderr: str, returncode: int) -> dict:
    try:
        data = json.loads(stdout)
    except ValueError:
        return {"ok": False, "text": "", "error": f"exit {returncode}: {(stderr or stdout).strip()[:300]}",
                "usage": _usage(), "usd_reported": None, "tool_events": 0}
    raw = data.get("usage") or {}
    usage = _usage(raw.get("input_tokens", 0) + raw.get("cache_creation_input_tokens", 0)
                   + raw.get("cache_read_input_tokens", 0), raw.get("cache_read_input_tokens", 0),
                   raw.get("output_tokens", 0), (raw.get("output_tokens_details") or {}).get("thinking_tokens", 0))
    result = data.get("result")
    failed = bool(data.get("is_error")) or not isinstance(result, str)
    served = sorted({str((entry or {}).get("canonicalModel") or name)
                     for name, entry in (data.get("modelUsage") or {}).items()})
    return {"ok": not failed, "text": "" if failed else result,
            "error": str(result)[:300] if failed else None, "usage": usage,
            "usd_reported": data.get("total_cost_usd"), "tool_events": 0, "served": served}


def parse_codex(stdout: str, stderr: str, returncode: int) -> dict:
    text, errors, tools, usage = "", [], 0, None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind, item = event.get("type"), event.get("item") or {}
        if kind == "item.completed":
            if item.get("type") == "agent_message":
                text = str(item.get("text", ""))
            elif item.get("type") == "error":
                errors.append(str(item.get("message", ""))[:300])
            elif item.get("type") != "reasoning":
                tools += 1
        elif kind == "turn.completed":
            raw = event.get("usage") or {}
            usage = _usage(raw.get("input_tokens", 0), raw.get("cached_input_tokens", 0),
                           raw.get("output_tokens", 0), raw.get("reasoning_output_tokens", 0))
        elif kind in ("turn.failed", "error"):
            errors.append(json.dumps(event.get("error") or event.get("message"))[:300])
    ok = bool(text) and usage is not None
    error = None if ok else (errors[0] if errors else f"exit {returncode}: {stderr.strip()[-300:]}")
    return {"ok": ok, "text": text, "error": error, "usage": usage or _usage(),
            "usd_reported": None, "tool_events": tools}


def command_for(slot: Slot, system: str, scratch: Path) -> list[str]:
    if slot.provider == "claude":
        # No MCP servers and no tools: a call carries ~1k tokens of context instead of 40k-140k.
        effort = [] if slot.effort == NO_EFFORT else ["--effort", slot.effort]
        return ["claude", "-p", "--output-format", "json", "--model", slot.model, *effort,
                "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--system-prompt", system, "--tools", ""]
    instructions = scratch / f"instructions-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:16]}.md"
    if not instructions.exists():
        partial = instructions.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(system + "\n", encoding="utf-8")
        partial.replace(instructions)
    effort = [] if slot.effort == NO_EFFORT else ["-c", f'model_reasoning_effort="{slot.effort}"']
    command = ["codex", "exec", "-m", slot.model, *effort,
               "-c", f'model_instructions_file="{instructions}"', "-c", "project_doc_max_bytes=0",
               "-c", 'web_search="disabled"',
               # The installed-skills listing alone is ~6k input tokens on every call.
               "-c", "skills.include_instructions=false"]
    for feature in CODEX_OFF:
        command += ["--disable", feature]
    return command + ["--json", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                      "--ignore-user-config", "--ignore-rules", "-C", str(scratch / "cwd"), "-"]


# Children this process is waiting on. They stay in this process's group on purpose: a caller
# such as `dsco interop run` kills that whole group on timeout or cancel, and a child moved to a
# session of its own would outlive the run.
_LIVE: set = set()
_LIVE_LOCK = threading.Lock()


def terminate_children(sig: int = signal.SIGTERM) -> int:
    """Signal every CLI child still running; for signal handlers and shutdown. -> how many."""
    with _LIVE_LOCK:
        live = list(_LIVE)
    for proc in live:
        try:
            proc.send_signal(sig)
        except (OSError, ValueError):
            pass
    return len(live)


def _spawn(command: list[str], prompt: str, *, timeout: float, cwd, env) -> tuple[str, str, int]:
    """Run one CLI child to completion. On timeout it is terminated, then killed, and always reaped."""
    # UTF-8 whatever the caller's locale: under LANG=C the default would fail on the first accent.
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", cwd=cwd, env=env)
    with _LIVE_LOCK:
        _LIVE.add(proc)
    try:
        try:
            out, err = proc.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            raise
        return out, err, proc.returncode
    finally:
        with _LIVE_LOCK:
            _LIVE.discard(proc)


def _fake(slot: Slot, prompt: str, timeout: float) -> dict:
    """Test seam: a `fake` provider scripted by the JSON file named in PIVOT_FAKE_SCRIPT.

    slot id -> {"text": ..., "error": ..., "usd": ..., "sleep": seconds, "exec": [argv]}. `exec`
    runs a real child through the same spawn path as the CLIs, to exercise timeouts and cleanup.
    Without the variable the provider does not exist."""
    empty = {"ok": False, "text": "", "usage": _usage(), "usd_reported": None, "tool_events": 0}
    try:
        script = json.loads(Path(os.environ["PIVOT_FAKE_SCRIPT"]).read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError):
        return {**empty, "error": "unknown provider fake"}
    act = script.get(slot.id) or script.get("*") or {}
    if act.get("exec"):
        try:
            out, _, code = _spawn(list(act["exec"]), prompt, timeout=timeout, cwd=None, env=dict(os.environ))
        except subprocess.TimeoutExpired:
            return {**empty, "error": "timeout"}
        return {**empty, "ok": code == 0, "text": out.strip(), "error": None if code == 0 else f"exit {code}"}
    if act.get("sleep"):
        time.sleep(min(float(act["sleep"]), timeout))
        if float(act["sleep"]) >= timeout:
            return {**empty, "error": "timeout"}
    if act.get("error"):
        return {**empty, "error": str(act["error"])}
    text = str(act.get("text", "")).replace("{prompt}", prompt)
    return {"ok": True, "text": text, "error": None, "usd_reported": float(act.get("usd", 0.001)),
            "usage": _usage(len(prompt) // 4, 0, max(1, len(text) // 4)), "tool_events": 0}


def same_model(asked: str, served: str) -> bool:
    return asked == served or served.startswith(asked) or asked.startswith(served)


def run_slot(slot: Slot, prompt: str, *, system: str = SYSTEM, timeout: float = 300.0) -> dict:
    """One stateless call. Never raises: failures and timeouts come back as receipts."""
    base = {"slot": slot.id, "provider": slot.provider, "model": slot.model, "effort": slot.effort}
    started = time.perf_counter()
    if slot.metered:
        # Billed per token: a direct HTTP call that uses the gateway's own key and records the price paid.
        parsed = gateways.call(slot.provider, slot.model, slot.effort, system, prompt, timeout)
        return {**base, **parsed, "price": slot.price_table(), "wall_s": round(time.perf_counter() - started, 2)}
    if slot.provider == "fake":
        return {**base, **_fake(slot, prompt, timeout), "wall_s": round(time.perf_counter() - started, 2)}
    if slot.provider not in FALLBACK_EFFORTS:
        return {**base, "ok": False, "text": "", "error": f"unknown provider {slot.provider}", "usage": _usage(),
                "usd_reported": None, "tool_events": 0, "wall_s": 0.0}
    env = {key: value for key, value in os.environ.items() if key not in METERED_KEYS + OVERRIDES}
    try:
        (SCRATCH / "cwd").mkdir(parents=True, exist_ok=True)
        stdout, stderr, code = _spawn(command_for(slot, system, SCRATCH), prompt, timeout=timeout,
                                      cwd=SCRATCH / "cwd", env=env)
        parse = parse_claude if slot.provider == "claude" else parse_codex
        parsed = parse(stdout, stderr, code)
        served = parsed.get("served") or []
        if parsed["ok"] and served and not any(same_model(slot.model, name) for name in served):
            # An answer from some other model must never be filed under this slot.
            parsed.update(ok=False, text="", error=f"served by {', '.join(served)}, not {slot.model}")
    except subprocess.TimeoutExpired:
        parsed = {"ok": False, "text": "", "error": "timeout", "usage": _usage(),
                  "usd_reported": None, "tool_events": 0}
    except OSError as error:
        parsed = {"ok": False, "text": "", "error": f"spawn: {error}", "usage": _usage(),
                  "usd_reported": None, "tool_events": 0}
    return {**base, **parsed, "wall_s": round(time.perf_counter() - started, 2)}
