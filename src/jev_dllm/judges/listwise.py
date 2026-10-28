"""Listwise Jev Choice over nested candidate subsets, keeping raw candidate text.

Wraps typesafe.build_request and typesafe.parse_response unchanged; the only
relaxation is one flagged two-decimal rounding step for the chosen option.
"""
from dataclasses import replace
import math
import random
from ..reproducibility import trajectory_seed
from ..schema import Candidate, Question
from .pointwise import post
from .typesafe import build_request, parse_response

ROUNDING_STEP = 0.01
INCONSISTENT = "Jev choice is inconsistent with its probabilities"


def parse_listwise_response(response: dict, request: dict) -> dict:
    """typesafe.parse_response, except that a choice at most one rounding step below the top
    probability is accepted when all probabilities have two decimals; it is flagged rounding_tie.

    Live responses report probabilities rounded to 0.01, so near-ties can show the chosen option
    0.01 below another. Every other check, and any larger gap, still rejects the response.
    """
    try:
        return {**parse_response(response, request), "rounding_tie": False}
    except ValueError as exc:
        if str(exc) != INCONSISTENT:
            raise
    # parse_response checked the model, choice, option set, values and sum before this check.
    answer = response["answers"]["best_answer"]
    probabilities = answer["probabilities"]
    two_decimal = all(math.isclose(p * 100, round(p * 100), abs_tol=1e-8) for p in probabilities.values())
    if not two_decimal or probabilities[answer["choice"]] < max(probabilities.values()) - ROUNDING_STEP - 1e-8:
        raise ValueError(INCONSISTENT + " beyond one two-decimal rounding step")
    confidence = answer["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid Jev confidence")
    usage = response["usage"]
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
        raise ValueError("Invalid Jev token usage")
    total = sum(probabilities.values())
    return {"candidate_id": answer["choice"], "probabilities": probabilities, "confidence": confidence,
            "model": response["model"], "usage": usage, "probability_sum": total,
            "probability_sum_tolerance": max(1e-4, len(probabilities) * 0.005 + 1e-8), "rounding_tie": True}


class ListwiseJudge:
    def __init__(self, timeout: float = 60):
        self.timeout = timeout

    def select(self, request: dict) -> dict:
        return post(request, parse_listwise_response, self.timeout)


def presentation_order(candidate_ids: list[str], seed: int, question_id: str) -> list[str]:
    """Deterministic shuffle of one subset, seeded as experiments/selection.py does."""
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate candidate IDs")
    order = list(candidate_ids)
    random.Random(trajectory_seed(seed, question_id, 0)).shuffle(order)
    return order


def nested_subsets(ordered_ids: list[str], sizes: tuple[int, ...]) -> dict[int, list[str]]:
    """Subset N is the first N IDs of one fixed order, so smaller subsets nest in larger ones."""
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Duplicate candidate IDs")
    if (not sizes or any(type(n) is not int for n in sizes) or list(sizes) != sorted(set(sizes))
            or sizes[0] < 2 or sizes[-1] > len(ordered_ids)):
        raise ValueError("Subset sizes must be increasing integers from 2 to the pool size")
    return {n: list(ordered_ids[:n]) for n in sizes}


def build_listwise_request(question: Question, candidates: list[Candidate], instructions: str,
                           seed: int, model: str = "jev-1.13.0") -> dict:
    """candidates is one subset; it is shuffled here before serialization."""
    by_id = {c.candidate_id: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    if any(not isinstance(c.text, str) for c in candidates):
        raise ValueError("Candidate text must be a string")
    order = presentation_order([c.candidate_id for c in candidates], seed, question.question_id)
    # final_request serializes only candidate_id and text, but rejects truncated or
    # incomplete answers. Neutralizing the unsent flags keeps such answers verbatim.
    raw = [replace(by_id[cid], completed=True, truncated=False) for cid in order]
    return build_request(question, raw, instructions, model)
