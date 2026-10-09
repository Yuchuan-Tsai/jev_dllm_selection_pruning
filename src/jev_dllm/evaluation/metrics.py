from statistics import mean


def pool_metrics(texts_by_question: dict[str, list[str]], labels_by_question: dict[str, list[bool]]) -> dict:
    if not texts_by_question or texts_by_question.keys() != labels_by_question.keys():
        raise ValueError("Require matching, nonempty question pools")
    for key, texts in texts_by_question.items():
        labels = labels_by_question[key]
        if not texts or len(texts) != len(labels) or any(type(v) is not bool for v in labels):
            raise ValueError(f"Invalid pool for {key}")
    return {
        "question_count": len(texts_by_question),
        "candidate_counts": {k: len(v) for k, v in texts_by_question.items()},
        "mean_single_sample_accuracy": mean(mean(v) for v in labels_by_question.values()),
        "empirical_oracle_at_pool_size": mean(any(v) for v in labels_by_question.values()),
        "mean_unique_text_fraction": mean(len(set(v)) / len(v) for v in texts_by_question.values()),
        "unit_of_analysis": "question",
    }


def gap_closed(selected: float, single: float, oracle: float) -> float | None:
    return None if oracle == single else (selected - single) / (oracle - single)
