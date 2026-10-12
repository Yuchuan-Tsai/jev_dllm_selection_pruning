"""TypeSafe Choice adapter. Only select() performs a paid network request.

Contract: https://docs.typesafe.ai/api (reviewed 2026-10-08).
"""
import json
import math
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from ..schema import Candidate, Question
from .payloads import final_request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def build_request(question: Question, candidates: list[Candidate], instructions: str,
                  model: str = "jev-1.13.0") -> dict:
    public = final_request(question, candidates)
    if not 2 <= len(candidates) <= 255:
        raise ValueError("Jev Choice requires 2 to 255 candidates in this experiment")
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
        raise ValueError("Pin a versioned Jev model; aliases are not reproducible")
    if not instructions.strip():
        raise ValueError("Selection instructions cannot be empty")
    return {"model": model, "state": public,
            "questions": {"best_answer": {"type": "choice", "instructions": instructions,
                "criteria": {c.candidate_id: f"Select candidate {c.candidate_id} from state.candidates."
                             for c in candidates}}}}


def parse_response(response: dict, request: dict) -> dict:
    if response.get("model") != request["model"]:
        raise ValueError("Jev returned a different model version")
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
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
        raise ValueError("Invalid Jev token usage")
    return {"candidate_id": answer["choice"], "probabilities": probabilities,
            "confidence": confidence, "model": response["model"], "usage": usage,
            "probability_sum": total, "probability_sum_tolerance": tolerance}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TypeSafeJudge:
    def __init__(self, timeout: float = 60):
        self.timeout = timeout

    def select(self, request: dict) -> dict:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY in the environment; .env is not loaded automatically")
        http = Request(ENDPOINT, data=json.dumps(request).encode(), method="POST",
                       headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with build_opener(_NoRedirect).open(http, timeout=self.timeout) as response:
                raw = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Jev HTTP {exc.code}; no automatic retry or fallback; billing may be unknown") from None
        except (URLError, TimeoutError) as exc:
            raise RuntimeError("Jev connection failed; no automatic retry; billing may be unknown") from None
        elapsed = time.perf_counter() - started
        try:
            result = parse_response(raw, request)
        except (ValueError, KeyError, TypeError) as exc:
            # Retain the returned public response for auditing a rejected API call.
            exc.raw_response = raw
            exc.request_seconds = elapsed
            raise
        result.update({"request_seconds": elapsed,
                       "api_cost_usd": None, "raw_response": raw})
        return result
