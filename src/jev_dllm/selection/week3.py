"""Week 3 selectors over one nested subset; every rule breaks ties by pool order.

All selectors are reference-free: they see candidate IDs, public answer keys and
scores. Correctness labels enter only the evaluation that grades their picks.
"""
import random

from ..reproducibility import trajectory_seed


def first(ids: list[str]) -> str:
    return ids[0]


def seeded_random(question_id: str, ids: list[str], seed: int) -> str:
    # Same stream index as Week 1's seeded_random, so N=8 picks are comparable in kind.
    return random.Random(trajectory_seed(seed, question_id, 1)).choice(ids)


def highest(ids: list[str], scores: dict) -> str:
    """Highest score; a missing score ranks last; ties go to the earliest candidate."""
    best = None
    for cid in ids:
        value = scores.get(cid)
        if value is not None and (best is None or value > scores[best]):
            best = cid
    return best if best is not None else ids[0]


def groups(ids: list[str], keys: dict) -> list[list[str]]:
    """Group candidates sharing an answer key, in order of first appearance; None keys do not vote."""
    by_key = {}
    for cid in ids:
        key = keys.get(cid)
        if key is not None:
            by_key.setdefault(key, []).append(cid)
    return list(by_key.values())


def majority(ids: list[str], keys: dict) -> str:
    """Largest answer group; ties go to the group seen first; no votes falls back to the first candidate."""
    found = groups(ids, keys)
    if not found:
        return ids[0]
    return max(found, key=len)[0]  # max keeps the earliest of equal-length groups.


def weighted_vote(ids: list[str], keys: dict, weights: dict) -> str:
    """Answer group with the largest summed weight (missing weights count 0); ties as in majority."""
    found = groups(ids, keys)
    if not found:
        return ids[0]
    return max(found, key=lambda g: sum(weights.get(cid) or 0.0 for cid in g))[0]


def prefix_group_keys(pool_ids: list[str], full_groups: list[list[int]]) -> dict:
    """Answer keys from complete-link groups computed in pool order over the whole pool.

    Greedy grouping in pool order only ever looks at earlier candidates, so the groups
    of a prefix subset are exactly the full-pool groups restricted to that prefix.
    """
    keys = {}
    for number, members in enumerate(full_groups):
        for index in members:
            keys[pool_ids[index]] = number
    return keys
