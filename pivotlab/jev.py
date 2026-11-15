"""TypeSafe Jev (System One) client: fan-out questions, strict validation, run ledger.

Stdlib only, so this package has no dependency outside itself.

Jev answers bounded questions; it never generates text and never holds authority.
Every answer is validated against the request before anything downstream reads it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.14.0"
# Published list price (https://docs.typesafe.ai/models): $0.042 per million input tokens; output free.
INPUT_USD_PER_MTOK = 0.042
# The live API rounds probabilities to hundredths; recorded sums as low as 0.99.
PROBABILITY_TOLERANCE = 0.021
SCORE_TOLERANCE = 0.041
RETRYABLE = {429, 500, 502, 503, 504}


class JevError(RuntimeError):
    """A Jev request failed, was refused locally, or returned a malformed answer."""


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def score(instructions: str, levels: list[str]) -> dict:
    if not 2 <= len(levels) <= 10:
        raise ValueError("score needs 2..10 ordered levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def choice(instructions: str, criteria: Mapping[str, Any]) -> dict:
    if not 2 <= len(criteria) <= 255:
        raise ValueError("choice needs 2..255 options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def _unit(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise JevError(f"jev_invalid_{what}")
    return float(value)


def validate_answers(questions: Mapping[str, dict], answers: Any) -> dict[str, dict]:
    """Return normalized answers covering every question; raise on any malformed answer."""
    if not isinstance(answers, Mapping):
        raise JevError("jev_answers_not_object")
    checked: dict[str, dict] = {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        kind = question["type"]
        if not isinstance(answer, Mapping) or answer.get("type") != kind:
            raise JevError(f"jev_answer_missing_or_mistyped:{qid}")
        if kind == "noul":
            checked[qid] = {"type": "noul", "noul": _unit(answer.get("noul"), "noul")}
            continue
        probs = answer.get("probabilities")
        if kind == "score":
            top = len(question["criteria"]) - 1
            expected = {str(i) for i in range(top + 1)}
        else:
            top = None
            expected = set(question["criteria"])
        if not isinstance(probs, Mapping) or set(probs) != expected:
            raise JevError(f"jev_probabilities_shape:{qid}")
        probs = {k: _unit(v, "probability") for k, v in probs.items()}
        if abs(sum(probs.values()) - 1) > PROBABILITY_TOLERANCE:
            raise JevError(f"jev_distribution:{qid}")
        confidence = _unit(answer.get("confidence"), "confidence")
        if kind == "score":
            value = answer.get("score")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= top:
                raise JevError(f"jev_score_range:{qid}")
            weighted = sum(int(k) * v for k, v in probs.items())
            if abs(weighted - value) > SCORE_TOLERANCE:
                raise JevError(f"jev_score_inconsistent:{qid}")
            checked[qid] = {"type": "score", "score": float(value), "top": top,
                            "normalized": float(value) / top, "confidence": confidence,
                            "probabilities": probs}
        else:
            selected = answer.get("choice")
            if selected not in expected:
                raise JevError(f"jev_choice_unknown:{qid}")
            checked[qid] = {"type": "choice", "choice": selected, "confidence": confidence,
                            "probabilities": probs}
    return checked


@dataclass
class Ledger:
    attempts: int = 0
    request_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    receipts: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {"attempts": self.attempts, "request_bytes": self.request_bytes,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "estimated_input_usd": round(self.input_tokens * INPUT_USD_PER_MTOK / 1_000_000, 6),
                "price_source": "https://docs.typesafe.ai/models", "billing_receipt": False}


Transport = Callable[[bytes], tuple[int, bytes]]


def _http_transport(endpoint: str, api_key: str, timeout: float) -> Transport:
    def send(body: bytes) -> tuple[int, bytes]:
        request = urllib.request.Request(endpoint, data=body, method="POST", headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, b""
    return send


class JevClient:
    """One process-local budget; no credential ever enters state, receipts or errors."""

    def __init__(self, *, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 endpoint: str = ENDPOINT, timeout: float = 60.0, max_attempts: int = 48,
                 max_request_bytes: int = 200_000, retries: int = 2,
                 transport: Transport | None = None) -> None:
        key = (api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")).strip()
        if transport is None and not key:
            raise JevError("jev_credential_missing: set TYPESAFE_API_KEY")
        self.model = model
        self.max_attempts = max_attempts
        self.max_request_bytes = max_request_bytes
        self.retries = retries
        self.ledger = Ledger()
        self._send = transport or _http_transport(endpoint, key, timeout)

    def ask(self, state: Any, questions: Mapping[str, dict]) -> dict:
        if not questions:
            raise JevError("jev_no_questions")
        body = json.dumps({"model": self.model, "state": state, "questions": dict(questions)},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(body) > self.max_request_bytes:
            raise JevError(f"jev_request_too_large:{len(body)}")
        digest = hashlib.sha256(body).hexdigest()
        for attempt in range(self.retries + 1):
            if self.ledger.attempts >= self.max_attempts:
                raise JevError("jev_attempt_limit")
            self.ledger.attempts += 1
            self.ledger.request_bytes += len(body)
            started = time.perf_counter()
            try:
                status, raw = self._send(body)
            except (urllib.error.URLError, TimeoutError, OSError):
                status, raw = 0, b""
            receipt = {"request_sha256": digest, "request_bytes": len(body), "status": status,
                       "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                       "attempt": attempt + 1, "question_count": len(questions)}
            if status == 200:
                try:
                    payload = json.loads(raw)
                except ValueError as error:
                    self.ledger.receipts.append(receipt)
                    raise JevError("jev_invalid_json") from error
                usage = payload.get("usage") if isinstance(payload, Mapping) else None
                usage = dict(usage) if isinstance(usage, Mapping) else {}
                for key_ in ("input_tokens", "output_tokens"):
                    tokens = usage.get(key_)
                    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                        setattr(self.ledger, key_, getattr(self.ledger, key_) + tokens)
                receipt.update(model=payload.get("model"), usage=usage)
                self.ledger.receipts.append(receipt)
                answers = validate_answers(questions, payload.get("answers"))
                return {"model": payload.get("model") or self.model, "answers": answers,
                        "usage": usage, "request_sha256": digest,
                        "latency_ms": receipt["latency_ms"]}
            self.ledger.receipts.append(receipt)
            retryable = status in RETRYABLE or status == 0
            if retryable and attempt < self.retries:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise JevError(f"jev_http_{status or 'transport'}")
        raise JevError("jev_unreachable")
