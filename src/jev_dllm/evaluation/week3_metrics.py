"""Week 3 selector metrics; the question is the unit of analysis throughout."""
from statistics import mean

from .metrics import gap_closed
from ..experiments.selection import paired_summary


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """P(random correct candidate outscores random incorrect one); ties count half."""
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def within_question_ranking(score_by_candidate: dict, pools: dict, correct: dict) -> dict:
    """Mean per-question AUROC over questions holding both correct and incorrect candidates.

    Missing scores rank lowest, as they do for the selector. Pooled AUROC is supplementary only.
    """
    per_question, pooled_scores, pooled_labels = [], [], []
    floor = min((v for v in score_by_candidate.values() if v is not None), default=0.0) - 1.0
    for qid, ids in pools.items():
        scores = [score_by_candidate.get(cid) if score_by_candidate.get(cid) is not None else floor for cid in ids]
        labels = [correct[cid] for cid in ids]
        value = auroc(scores, labels)
        if value is not None:
            per_question.append(value)
        pooled_scores += scores
        pooled_labels += labels
    return {"mean_within_question_auroc": mean(per_question) if per_question else None,
            "mixed_questions": len(per_question), "pooled_auroc_supplementary": auroc(pooled_scores, pooled_labels)}


def calibration(probability_by_candidate: dict, correct: dict, bins: int = 10) -> dict:
    """Brier score and equal-width reliability bins for a score that predicts correctness."""
    pairs = [(p, correct[cid]) for cid, p in probability_by_candidate.items() if p is not None]
    if not pairs:
        return {"count": 0}
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if inside:
            table.append({"bin": [lo, hi], "count": len(inside), "mean_predicted": mean(p for p, _ in inside),
                          "observed_accuracy": mean(y for _, y in inside)})
    ece = sum(row["count"] * abs(row["mean_predicted"] - row["observed_accuracy"]) for row in table) / len(pairs)
    return {"count": len(pairs), "brier": mean((p - y) ** 2 for p, y in pairs),
            "expected_calibration_error": ece, "reliability_bins": table}


def failure_tags(selected: dict, subsets: dict, correct: dict) -> dict:
    """Split misses into pool failures (nothing correct to pick) and selector failures."""
    tags = {"correct": 0, "pool_failure": 0, "selector_failure": 0}
    for qid, cid in selected.items():
        if correct[cid]:
            tags["correct"] += 1
        elif any(correct[i] for i in subsets[qid]):
            tags["selector_failure"] += 1
        else:
            tags["pool_failure"] += 1
    return tags


def compare(selections: dict, subsets: dict, correct: dict, seed: int, baselines=("seeded_random", "majority")) -> dict:
    """selections[name][qid] = chosen ID within subsets[qid]; returns accuracy, gap closed and paired CIs."""
    qids = sorted(subsets)
    single = mean(mean(correct[c] for c in subsets[q]) for q in qids)
    oracle = mean(any(correct[c] for c in subsets[q]) for q in qids)
    outcomes = {name: [correct[chosen[q]] for q in qids] for name, chosen in selections.items()}
    result = {"questions": len(qids), "mean_single_sample_accuracy": single, "oracle_accuracy": oracle, "selectors": {}}
    for name, values in outcomes.items():
        accuracy = mean(values)
        entry = {"accuracy": accuracy, "correct": sum(values),
                 "gap_closed": gap_closed(accuracy, single, oracle),
                 "failures": failure_tags(selections[name], subsets, correct)}
        for base in baselines:
            if base in outcomes and base != name:
                entry[f"vs_{base}"] = paired_summary(values, outcomes[base], seed)
        result["selectors"][name] = entry
    return result
