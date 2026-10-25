"""Jev AI Choice adapter. Only select() performs a paid network request.

Contract: https://jev-ai.org/jev-api/.
"""
import hashlib
import json
import math
import os
import time
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from ..schema import Candidate, Question
from .payloads import final_request

ENDPOINT = "https://jev-ai.org/api/v1/systemone/"
DEFAULT_MODEL = "jev-1.13"
RETRYABLE_STATUS = {429, 502, 503, 504}
MAX_ATTEMPTS = 4


def build_request(question: Question, candidates: list[Candidate], instructions: str,
                  model: str = DEFAULT_MODEL) -> dict:
    public = final_request(question, candidates)
    if not 2 <= len(candidates) <= 255:
        raise ValueError("Jev Choice requires 2 to 255 candidates in this experiment")
    if model not in {"jev-1.13", "jev-latest"}:
        raise ValueError("Jev AI supports jev-1.13 or jev-latest")
    if not instructions.strip():
        raise ValueError("Selection instructions cannot be empty")
    return {"model": model, "state": public,
            "questions": {"best_answer": {"type": "choice", "instructions": instructions,
                "criteria": {c.candidate_id: f"Select candidate {c.candidate_id} from state.candidates."
                             for c in candidates}}}}


def parse_response(response: dict, request: dict) -> dict:
    if response.get("model") != request["model"]:
        raise ValueError("Jev returned a different model alias")
    answer = response["answers"]["best_answer"]
    expected = set(request["questions"]["best_answer"]["criteria"])
    probabilities = answer["probabilities"]
    if answer.get("type") != "choice" or answer.get("choice") not in expected:
        raise ValueError("Jev returned an invalid candidate choice")
    if set(probabilities) != expected:
        raise ValueError("Jev probabilities do not match the submitted candidate IDs")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities.values()):
        raise ValueError("Invalid Jev probability distribution")
    total = sum(probabilities.values())
    # Observed live API values use two decimal places. Independent rounding can
    # change the sum by at most 0.005 per option. Preserve raw values, never normalize.
    two_decimal_values = all(math.isclose(p * 100, round(p * 100), abs_tol=1e-8) for p in probabilities.values())
    tolerance = max(1e-4, len(probabilities) * 0.005 + 1e-8) if two_decimal_values else 1e-4
    if total <= 0 or not math.isclose(total, 1, rel_tol=0, abs_tol=tolerance):
        raise ValueError("Invalid Jev probability distribution")
    if probabilities[answer["choice"]] < max(probabilities.values()) - 1e-8:
        raise ValueError("Jev choice is inconsistent with its probabilities")
    confidence = answer["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid Jev confidence")
    usage = response["usage"]
    if any(type(usage.get(k)) is not int or usage[k] < 0
           for k in ("input_tokens", "output_tokens", "charged_tokens", "charged_credits")):
        raise ValueError("Invalid Jev token usage")
    return {"candidate_id": answer["choice"], "probabilities": probabilities,
            "confidence": confidence, "model": response["model"],
            "model_version": response.get("model_version"), "decision_id": response.get("id"),
            "usage": usage,
            "probability_sum": total, "probability_sum_tolerance": tolerance}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TypeSafeJudge:
    """Compatibility class name retained for existing workflow imports."""

    def __init__(self, timeout: float = 60, max_attempts: int = MAX_ATTEMPTS):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.timeout = timeout
        self.max_attempts = max_attempts

    @staticmethod
    def _retry_delay(exc: HTTPError, attempt: int) -> float:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            try:
                return max(0.0, min(30.0, float(retry_after)))
            except ValueError:
                try:
                    wait = (parsedate_to_datetime(retry_after) - parsedate_to_datetime(
                        time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()))).total_seconds()
                    return max(0.0, min(30.0, wait))
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(8.0, 0.5 * (2 ** (attempt - 1)))

    def select(self, request: dict) -> dict:
        key = os.environ.get("JEV_API_KEY")
        if not key:
            raise ValueError("Set JEV_API_KEY in the environment; .env is not loaded automatically")
        payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        idempotency_key = hashlib.sha256(payload).hexdigest()
        started = time.perf_counter()
        raw = None
        attempts = 0
        for attempts in range(1, self.max_attempts + 1):
            http = Request(ENDPOINT, data=payload, method="POST", headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key})
            try:
                with build_opener(_NoRedirect).open(http, timeout=self.timeout) as response:
                    raw = json.load(response)
                break
            except HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and attempts < self.max_attempts:
                    time.sleep(self._retry_delay(exc, attempts))
                    continue
                error_code = None
                try:
                    error_body = json.loads(exc.read().decode("utf-8"))
                    error_code = error_body.get("error", {}).get("code")
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                    pass
                error = RuntimeError(
                    f"Jev AI HTTP {exc.code} (error.code={error_code or 'unknown'}); "
                    f"attempts={attempts}; inspect account/request; no fallback"
                )
                error.api_attempts = attempts
                raise error from None
            except (URLError, TimeoutError) as exc:
                if attempts < self.max_attempts:
                    time.sleep(min(8.0, 0.5 * (2 ** (attempts - 1))))
                    continue
                error = RuntimeError(
                    f"Jev AI connection failed after {attempts} idempotent attempts; "
                    "billing status may be unknown"
                )
                error.api_attempts = attempts
                raise error from None
        elapsed = time.perf_counter() - started
        try:
            result = parse_response(raw, request)
        except (ValueError, KeyError, TypeError) as exc:
            # Retain the returned public response for auditing a rejected API call.
            exc.raw_response = raw
            exc.request_seconds = elapsed
            exc.api_attempts = attempts
            raise
        result.update({"request_seconds": elapsed, "api_attempts": attempts,
                       "api_cost_usd": None, "idempotency_key": idempotency_key,
                       "raw_response": raw})
        return result


JevAIJudge = TypeSafeJudge
