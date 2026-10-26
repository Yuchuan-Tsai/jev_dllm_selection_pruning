"""TypeSafe pointwise judging: one candidate per paid request, two separate noul targets.

Contract: https://docs.typesafe.ai/api (reviewed 2026-10-26). A noul answer is
{"type": "noul", "noul": p}, where p in [0, 1] is the probability of yes/true.
Only judge() performs a paid network request.
"""
import json
import math
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
from ..schema import Candidate, Question
from .typesafe import ENDPOINT, _NoRedirect

# Correctness and reasoning consistency are different targets; never merge them.
TARGETS = ("correct", "consistent")


def pointwise_state(question: Question, candidate: Candidate) -> dict:
    """Explicit allowlist: question text plus one candidate's ID and verbatim text."""
    if candidate.question_id != question.question_id:
        raise ValueError("Candidate belongs to a different question")
    if not isinstance(candidate.candidate_id, str) or not candidate.candidate_id:
        raise ValueError("Candidate ID must be a nonempty string")
    if not isinstance(candidate.text, str) or not isinstance(question.prompt, str) or not question.prompt.strip():
        raise ValueError("Question and candidate text must be strings")
    # Truncated, incomplete and empty answers are kept verbatim; flags are never sent.
    return {"stage": "pointwise", "question": question.prompt,
            "candidate": {"candidate_id": candidate.candidate_id, "text": candidate.text}}


def build_pointwise_request(question: Question, candidate: Candidate, correct_instructions: str,
                            consistent_instructions: str, model: str = "jev-1.13.0") -> dict:
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
        raise ValueError("Pin a versioned Jev model; aliases are not reproducible")
    instructions = {"correct": correct_instructions, "consistent": consistent_instructions}
    if any(not isinstance(v, str) or not v.strip() for v in instructions.values()):
        raise ValueError("Pointwise instructions cannot be empty")
    return {"model": model, "state": pointwise_state(question, candidate),
            "questions": {name: {"type": "noul", "instructions": instructions[name]} for name in TARGETS}}


def parse_pointwise_response(response: dict, request: dict) -> dict:
    model = response.get("model")
    if not isinstance(model, str) or not re.fullmatch(r"jev-\d+\.\d+\.\d+", model) or model != request["model"]:
        raise ValueError("Jev returned a different model version")
    answers = response["answers"]
    if not isinstance(answers, dict) or set(answers) != set(TARGETS):
        raise ValueError("Jev pointwise answers do not match the submitted questions")
    values = {}
    for name in TARGETS:
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError(f"Jev answer {name} is not a noul answer")
        value = answer["noul"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid Jev noul probability for {name}")
        values[name] = value
    usage = response["usage"]
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
        raise ValueError("Invalid Jev token usage")
    return {"candidate_id": request["state"]["candidate"]["candidate_id"],
            "correct_probability": values["correct"], "consistent_probability": values["consistent"],
            "model": model, "usage": usage}


def post(request: dict, parse, timeout: float) -> dict:
    """One POST: no redirects, no automatic retry; the raw response is kept on rejection."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError("Set TYPESAFE_API_KEY in the environment; .env is not loaded automatically")
    http = Request(ENDPOINT, data=json.dumps(request).encode(), method="POST",
                   headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with build_opener(_NoRedirect).open(http, timeout=timeout) as response:
            raw = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Jev HTTP {exc.code}; no automatic retry or fallback; billing may be unknown") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Jev connection failed; no automatic retry; billing may be unknown") from None
    elapsed = time.perf_counter() - started
    try:
        result = parse(raw, request)
    except (ValueError, KeyError, TypeError) as exc:
        exc.raw_response = raw
        exc.request_seconds = elapsed
        raise
    result.update({"request_seconds": elapsed, "api_cost_usd": None, "raw_response": raw})
    return result


class PointwiseJudge:
    def __init__(self, timeout: float = 60):
        self.timeout = timeout

    def judge(self, request: dict) -> dict:
        return post(request, parse_pointwise_response, self.timeout)
