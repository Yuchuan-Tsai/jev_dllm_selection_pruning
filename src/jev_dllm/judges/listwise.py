"""Listwise Jev Choice over nested candidate subsets, keeping raw candidate text.

Wraps typesafe.build_request and typesafe.parse_response unchanged.
"""
from dataclasses import replace
import random
from ..reproducibility import trajectory_seed
from ..schema import Candidate, Question
from .typesafe import build_request, parse_response

parse_listwise_response = parse_response


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
