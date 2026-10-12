"""Reference-free selectors over a fixed, ordered candidate pool."""
from collections import Counter
import random
from ..evaluation.answers import numeric_answer
from ..schema import Candidate


def validate_pool(candidates: list[Candidate]) -> None:
    if not candidates or len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Require a nonempty pool with unique candidate IDs")
    if len({c.question_id for c in candidates}) != 1:
        raise ValueError("A pool must belong to one question")


def first(candidates: list[Candidate]) -> str:
    validate_pool(candidates)
    return candidates[0].candidate_id


def random_choice(candidates: list[Candidate], seed: int) -> str:
    validate_pool(candidates)
    return random.Random(seed).choice(candidates).candidate_id


def majority_numeric(candidates: list[Candidate], scorer: str = "gsm8k_strict_v1") -> str:
    """Count equivalent numeric answers; ties use original candidate order.

    Invalid/incomplete answers do not vote. If none can vote, choose the first
    candidate; offline evaluation counts its invalid answer as incorrect.
    """
    validate_pool(candidates)
    values = [numeric_answer(c.text, scorer)
              if c.completed and not c.truncated else None for c in candidates]
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return first(candidates)
    largest = max(counts.values())
    return next(c.candidate_id for c, v in zip(candidates, values)
                if v is not None and counts[v] == largest)
