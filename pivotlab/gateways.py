"""Metered gateway slots: OpenRouter (chat completions) and Ramp Router (OpenAI Responses API).

A second layer under the subscription CLIs: the same top models and every other vendor's, paid
per token. Because this is billed money, a model with no listed price is never called, every
call carries an output-token ceiling, and the grid reserves a call's worst-case cost against
the provider's cap before sending it (see grid.Budget).

Each gateway publishes its own model list with prices and the reasoning efforts a model really
accepts; only those efforts become slots. Both gateways map an unsupported effort to a
neighbour without saying so, which would make two slots the same slot.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

GATEWAYS = {
    "openrouter": {"base": "https://openrouter.ai/api/v1", "key": "OPENROUTER_API_KEY", "list_needs_key": False},
    "ramp": {"base": "https://api.router.com/v1", "key": "RAMP_ROUTER_API_KEY", "list_needs_key": True},
}
EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
CATALOG_TTL_S = 24 * 3600
MAX_OUTPUT_TOKENS = int(os.environ.get("PIVOT_MAX_OUTPUT_TOKENS") or 20000)


def _ordered(efforts) -> list[str]:
    known = [str(e) for e in efforts if str(e) in EFFORT_ORDER]
    return sorted(dict.fromkeys(known), key=EFFORT_ORDER.index)


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_openrouter_models(payload: dict) -> dict:
    """OpenRouter prices are dollars per token; a negative price marks a router alias, not a model."""
    out = {}
    for model in payload.get("data") or []:
        pricing = model.get("pricing") or {}
        prompt, completion = _number(pricing.get("prompt")), _number(pricing.get("completion"))
        if prompt is None or completion is None or prompt < 0 or completion < 0:
            continue
        cached = _number(pricing.get("input_cache_read"))
        reasoning = model.get("reasoning") or {}
        listed = _ordered(reasoning.get("supported_efforts") or [])
        if not listed and "reasoning" in (model.get("supported_parameters") or []):
            listed = ["low", "medium", "high"]
        modalities = (model.get("architecture") or {}).get("output_modalities") or ["text"]
        out[str(model["id"])] = {"efforts": listed, "created": model.get("created") or 0,
                                 "text": "text" in modalities, "price": {
            "input": round(prompt * 1e6, 6), "output": round(completion * 1e6, 6),
            "cached_input": round((prompt if cached is None else cached) * 1e6, 6)}}
    return out


def parse_ramp_models(payload: dict) -> dict:
    """Ramp Router prices are decimal strings in dollars per million tokens, under `router.pricing`."""
    out = {}
    for model in payload.get("data") or []:
        router = model.get("router") or {}
        pricing = router.get("pricing") or {}
        price_in, price_out = _number(pricing.get("input")), _number(pricing.get("output"))
        if price_in is None or price_out is None or "responses" not in (router.get("surfaces") or ["responses"]):
            continue
        cached = _number(pricing.get("cache_read_input"))
        reasoning = (router.get("capabilities") or {}).get("reasoning") or {}
        efforts = [e.get("value") if isinstance(e, dict) else e for e in reasoning.get("efforts") or []]
        out[str(model["id"])] = {"efforts": _ordered(efforts) if reasoning.get("supported") else [],
                                 "created": model.get("created") or 0, "text": True, "price": {
            "input": price_in, "output": price_out, "cached_input": price_in if cached is None else cached}}
    return out


PARSERS = {"openrouter": parse_openrouter_models, "ramp": parse_ramp_models}


def http_json(url: str, key: str | None = None, body: dict | None = None, timeout: float = 60.0):
    """-> (status, parsed body, headers). Never raises on an HTTP error status."""
    headers = {"Content-Type": "application/json", "User-Agent": "jev-pivot-lab"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            status, raw, got = reply.status, reply.read(), dict(reply.headers.items())
    except urllib.error.HTTPError as error:
        status, raw, got = error.code, error.read(), dict(error.headers.items())
    try:
        return status, json.loads(raw.decode("utf-8", "replace")), got
    except ValueError:
        return status, {"error": {"message": raw.decode("utf-8", "replace")[:300]}}, got


def catalog(provider: str, scratch: Path, fetch=http_json) -> dict:
    """model id -> {"efforts": [...], "price": {...}} as the gateway lists it; kept on disk for a day."""
    spec = GATEWAYS[provider]
    path = Path(scratch) / f"{provider}-models.json"
    if path.exists() and time.time() - path.stat().st_mtime < CATALOG_TTL_S:
        return PARSERS[provider](json.loads(path.read_text(encoding="utf-8")))
    key = os.environ.get(spec["key"])
    if spec["list_needs_key"] and not key:
        raise ValueError(f"{spec['key']} is not set; {provider} needs it even to list its models")
    status, payload, _ = fetch(f"{spec['base']}/models", key if spec["list_needs_key"] else None)
    if status != 200:
        raise ValueError(f"{provider} model list failed with HTTP {status}: {json.dumps(payload)[:200]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return PARSERS[provider](payload)


def worst_case_usd(price: dict, system: str, prompt: str) -> float:
    """An upper bound on one call: every character a token-third, and the full output ceiling used."""
    return ((len(system) + len(prompt)) / 3 * price["input"] + MAX_OUTPUT_TOKENS * price["output"]) / 1e6


def request_body(provider: str, model: str, effort: str, system: str, prompt: str) -> dict:
    reasoning = {} if effort == "default" else {"reasoning": {"effort": effort}}
    if provider == "openrouter":
        return {"model": model, "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}],
                "max_tokens": MAX_OUTPUT_TOKENS, "usage": {"include": True}, **reasoning}
    return {"model": model, "instructions": system, "input": prompt,
            "max_output_tokens": MAX_OUTPUT_TOKENS, **reasoning}


def parse_reply(provider: str, status: int, data: dict, headers: dict) -> dict:
    usage = {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
    failed = {"ok": False, "text": "", "usage": usage, "usd_reported": None, "tool_events": 0, "served": []}
    if status != 200 or not isinstance(data, dict) or data.get("error"):
        error = (data.get("error") if isinstance(data, dict) else None) or {}
        message = error.get("message") if isinstance(error, dict) else error
        return {**failed, "error": f"HTTP {status}: {str(message or data)[:240]}"}
    raw = data.get("usage") or {}
    if provider == "openrouter":
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = {"input": int(raw.get("prompt_tokens") or 0),
                 "cached_input": int((raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                 "output": int(raw.get("completion_tokens") or 0),
                 "reasoning": int((raw.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)}
    else:
        parts = [part.get("text") or "" for item in data.get("output") or [] if item.get("type") == "message"
                 for part in item.get("content") or [] if part.get("type") in ("output_text", "text")]
        text = "".join(parts)
        usage = {"input": int(raw.get("input_tokens") or 0),
                 "cached_input": int((raw.get("input_tokens_details") or {}).get("cached_tokens") or 0),
                 "output": int(raw.get("output_tokens") or 0),
                 "reasoning": int((raw.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)}
    served = [str(data["model"])] if data.get("model") else []
    upstream = data.get("provider") or headers.get("X-Router-Provider") or headers.get("x-router-provider")
    if not isinstance(text, str) or not text.strip():
        if not usage["output"]:
            return {**failed, "usage": usage, "served": served, "usd_reported": _number(raw.get("cost")),
                    "error": "empty reply with no usage"}
        # Tokens were produced and billed but no answer came out, usually because reasoning used the
        # whole output ceiling. That is a wrong answer the model paid for, not a gateway failure:
        # it is kept, priced and graded like any other reply.
        text = ""
    return {"ok": True, "text": text, "error": None, "usage": usage, "usd_reported": _number(raw.get("cost")),
            "tool_events": 0, "served": served, "upstream": upstream,
            "truncated": not text or usage["output"] >= MAX_OUTPUT_TOKENS}


def call(provider: str, model: str, effort: str, system: str, prompt: str, timeout: float, post=http_json) -> dict:
    """One metered call. Never raises: a missing key, a timeout or an HTTP error comes back as a receipt."""
    key = os.environ.get(GATEWAYS[provider]["key"])
    empty = {"ok": False, "text": "", "usage": {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0},
             "usd_reported": None, "tool_events": 0, "served": []}
    if not key:
        return {**empty, "error": f"{GATEWAYS[provider]['key']} is not set"}
    path = "/chat/completions" if provider == "openrouter" else "/responses"
    try:
        status, data, headers = post(GATEWAYS[provider]["base"] + path, key,
                                     request_body(provider, model, effort, system, prompt), timeout)
    except TimeoutError:
        return {**empty, "error": "timeout"}
    except (OSError, urllib.error.URLError) as error:
        timed_out = "timed out" in str(error).lower()
        return {**empty, "error": "timeout" if timed_out else f"network: {str(error)[:200]}"}
    return parse_reply(provider, status, data, headers)
