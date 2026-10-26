"""Fixed nested candidate subsets, so N=2/4/8/16 comparisons share their candidates."""
from ..reproducibility import fingerprint

SIZES = (2, 4, 8, 16)


def nested_subsets(question_id: str, candidate_ids: list[str], seed: int, sizes=SIZES) -> dict:
    """Return {N: candidate IDs}; each subset keeps original pool order and contains all smaller ones.

    Membership comes from one seeded permutation of the pool, never from text or labels.
    Original order inside a subset is the tie-break order every selector uses.
    """
    if len(set(candidate_ids)) != len(candidate_ids) or not candidate_ids:
        raise ValueError("Require a nonempty pool with unique candidate IDs")
    if list(sizes) != sorted(set(sizes)) or sizes[0] < 1 or sizes[-1] > len(candidate_ids):
        raise ValueError("Subset sizes must increase and fit inside the pool")
    permutation = sorted(candidate_ids, key=lambda cid: fingerprint([seed, "week02-nested-subsets", question_id, cid]))
    position = {cid: i for i, cid in enumerate(candidate_ids)}
    return {n: sorted(permutation[:n], key=position.__getitem__) for n in sizes}
