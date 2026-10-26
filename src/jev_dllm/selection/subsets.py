"""Fixed nested candidate subsets, so N=2/4/8/16 comparisons share their candidates."""

SIZES = (2, 4, 8, 16)


def nested_subsets(candidate_ids: list[str], sizes=SIZES) -> dict:
    """Return {N: the first N candidate IDs in pool order}.

    Candidate i is seeded by trajectory_seed(master_seed, question_id, i), so pool order
    is an exchangeable sample: a prefix is an unbiased subset that never depends on text
    or labels. Prefixes nest, keep Week 1's seeds for N <= 8, and pool order is the
    tie-break order every selector uses. week2_jev builds its requests from the same prefixes.
    """
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Require a nonempty pool with unique candidate IDs")
    if list(sizes) != sorted(set(sizes)) or sizes[0] < 1 or sizes[-1] > len(candidate_ids):
        raise ValueError("Subset sizes must increase and fit inside the pool")
    return {n: list(candidate_ids[:n]) for n in sizes}
