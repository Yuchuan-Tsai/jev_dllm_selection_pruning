"""Separate public judging from private evaluation of saved candidate pools."""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import hashlib
import math
from pathlib import Path
import random
from statistics import mean
import uuid
from ..evaluation.answers import numeric_value
from ..evaluation.metrics import pool_metrics, gap_closed
from ..judges.typesafe import TypeSafeJudge, build_request
from ..reproducibility import fingerprint, trajectory_seed
from ..schema import Candidate, Question
from ..selection.baselines import first, majority_numeric, random_choice, validate_pool


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def new_run(output: Path, prefix: str) -> Path:
    path = output / (datetime.now(timezone.utc).strftime(prefix + "-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_questions(path: Path) -> list[Question]:
    questions = [Question(**row) for row in json.loads(path.read_text())]
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Questions must have unique IDs and be nonempty")
    for q in questions:
        if q.split not in {"dev", "heldout", "fixture"} or not all(
                isinstance(v, str) and v.strip() for v in (q.question_id, q.dataset, q.prompt)):
            raise ValueError("Invalid public question record")
    return questions


def load_pools(questions_path: Path, candidates_path: Path) -> tuple[list[Question], dict]:
    questions = load_questions(questions_path)
    pools = defaultdict(list)
    for line in candidates_path.read_text().splitlines():
        if not line.strip():
            continue
        candidate = Candidate(**json.loads(line))
        if (type(candidate.completed) is not bool or type(candidate.truncated) is not bool
                or type(candidate.seed) is not int or candidate.seed < 0
                or not isinstance(candidate.text, str)
                or not isinstance(candidate.candidate_id, str) or not candidate.candidate_id
                or not isinstance(candidate.question_id, str)):
            raise ValueError("Invalid candidate record")
        if candidate.confidence is not None and (type(candidate.confidence) not in (int, float)
                                                 or not math.isfinite(candidate.confidence)):
            raise ValueError("Invalid candidate confidence")
        pools[candidate.question_id].append(candidate)
    if set(pools) != {q.question_id for q in questions}:
        raise ValueError("Candidate pools must cover exactly the supplied questions")
    for pool in pools.values():
        validate_pool(pool)
    if len({len(pool) for pool in pools.values()}) != 1:
        raise ValueError("Use the same candidate count for every question")
    return questions, dict(pools)


def judge_pools(questions_path: Path, candidates_path: Path, prompt_path: Path,
               output: Path, model: str, seed: int = 20261008, live: bool = False,
               judge=None, protocol: dict | None = None) -> Path:
    """No reference argument is accepted. Dry-run is the default."""
    questions, pools = load_pools(questions_path, candidates_path)
    instructions = prompt_path.read_text()
    requests = []
    for q in questions:
        order = list(pools[q.question_id])
        random.Random(trajectory_seed(seed, q.question_id, 0)).shuffle(order)
        requests.append({"question_id": q.question_id,
                         "body": build_request(q, order, instructions, model)})
    # Validate every pool before the first request can incur a charge.
    run_dir = new_run(output, "jev-live" if live else "jev-dry-run")
    (run_dir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    manifest = {"mode": "live_jev" if live else "dry_run", "status": "prepared",
                "question_count": len(questions), "attempted_calls": 0,
                "successful_calls": 0, "model": model, "master_seed": seed,
                "questions_sha256": fingerprint([asdict(q) for q in questions]),
                "pools_sha256": fingerprint({k: [asdict(c) for c in v] for k, v in pools.items()}),
                "prompt_sha256": fingerprint(instructions), "api_cost_usd": 0.0 if not live else None,
                "note": "Transient retries use a stable idempotency key; unknown billing remains null. Dry-run has no Jev result."}
    if protocol is not None:
        manifest["protocol"] = protocol
    write_json(run_dir / "manifest.json", manifest)
    if live:
        client = judge or TypeSafeJudge()
        try:
            with (run_dir / "selections.jsonl").open("w") as stream:
                for row in requests:
                    manifest["attempted_calls"] += 1
                    write_json(run_dir / "manifest.json", manifest)
                    result = client.select(row["body"])
                    manifest["attempted_calls"] += max(0, result.get("api_attempts", 1) - 1)
                    stream.write(json.dumps({"question_id": row["question_id"], **result}) + "\n")
                    stream.flush()
                    manifest["successful_calls"] += 1
        except Exception as exc:
            manifest["attempted_calls"] += max(0, getattr(exc, "api_attempts", 1) - 1)
            if hasattr(exc, "raw_response"):
                write_json(run_dir / "failed_response.json", {"question_id": row["question_id"],
                           "raw_response_json": json.dumps(exc.raw_response, ensure_ascii=False),
                           "request_seconds": exc.request_seconds})
            manifest.update({"status": "failed", "error_type": type(exc).__name__,
                             "failed_question_id": row["question_id"]})
            write_json(run_dir / "manifest.json", manifest)
            raise
        manifest["status"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
    return run_dir


def paired_summary(a: list[bool], b: list[bool], seed: int) -> dict:
    if not a or len(a) != len(b) or any(type(x) is not bool for x in a + b):
        raise ValueError("Paired outcomes must be nonempty boolean lists of equal length")
    differences = [int(x) - int(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
    return {"accuracy_difference": mean(differences), "wins": differences.count(1),
            "losses": differences.count(-1), "ties": differences.count(0),
            "bootstrap_95_percent_interval": [estimates[49], estimates[1949]],
            "bootstrap_replicates": 2000, "bootstrap_seed": seed,
            "unit": "question", "interval_method": "paired percentile bootstrap"}


def evaluate_pools(questions_path: Path, candidates_path: Path, references_path: Path,
                   output: Path, selections_path: Path | None = None,
                   seed: int = 20261008, scorer: str = "gsm8k_strict_v1",
                   protocol: dict | None = None) -> Path:
    from ..evaluation.answers import numeric_answer, SCORERS
    if scorer not in SCORERS:
        raise ValueError("Unknown numeric scoring version")
    questions, pools = load_pools(questions_path, candidates_path)
    if any(q.dataset != "gsm8k" and q.split != "fixture" for q in questions):
        raise ValueError("This numeric evaluator supports GSM8K only; MATH500 and HumanEval require their dedicated validated evaluators")
    references = json.loads(references_path.read_text())
    refs = {r["question_id"]: r["answer"] for r in references}
    if len(refs) != len(references) or set(refs) != set(pools):
        raise ValueError("References must cover exactly the supplied questions, without duplicates")
    if any(not isinstance(value, str) or numeric_value(value) is None for value in refs.values()):
        raise ValueError("All references must be valid numeric strings")
    selected = None
    if selections_path:
        manifest = json.loads((selections_path.parent / "manifest.json").read_text())
        if (manifest.get("status") != "complete" or manifest.get("mode") != "live_jev"
                or manifest.get("questions_sha256") != fingerprint([asdict(q) for q in questions])
                or manifest.get("pools_sha256") != fingerprint({k: [asdict(c) for c in v] for k, v in pools.items()})):
            raise ValueError("Jev results must come from a completed run on these exact questions and candidate pools")
        rows = [json.loads(line) for line in selections_path.read_text().splitlines() if line.strip()]
        selected = {r["question_id"]: r["candidate_id"] for r in rows}
        if len(selected) != len(rows) or set(selected) != set(pools):
            raise ValueError("Incomplete/duplicate Jev results: do not silently drop failed questions")
    outcomes = {name: [] for name in ("first_sample", "seeded_random", "majority_vote")}
    if selected is not None:
        outcomes["jev"] = []
    texts, labels, details = {}, {}, []
    for q in questions:
        pool = pools[q.question_id]
        correct = {c.candidate_id: bool(c.completed and not c.truncated and numeric_answer(c.text, scorer) == numeric_value(refs[q.question_id]))
                   for c in pool}
        # Selectors receive only public candidates. References enter above for offline scoring.
        ids = {"first_sample": first(pool), "seeded_random": random_choice(pool, trajectory_seed(seed, q.question_id, 1)),
               "majority_vote": majority_numeric(pool, scorer)}
        if selected is not None:
            ids["jev"] = selected[q.question_id]
        for method, candidate_id in ids.items():
            if candidate_id not in correct:
                raise ValueError(f"Unknown selected candidate for {q.question_id}")
            outcomes[method].append(correct[candidate_id])
        texts[q.question_id] = [c.text for c in pool]
        labels[q.question_id] = list(correct.values())
        details.append({"question_id": q.question_id, "selected_ids": ids, "correct_by_candidate": correct})
    stats = pool_metrics(texts, labels)
    all_candidates = [c for pool in pools.values() for c in pool]
    stats.update({"accuracy": {name: mean(values) for name, values in outcomes.items()},
                  "expected_uniform_random_accuracy": stats["mean_single_sample_accuracy"],
                  "jev_evaluated": selected is not None,
                  "incomplete_or_truncated_fraction": mean(not c.completed or c.truncated for c in all_candidates),
                  "unparseable_fraction": mean(numeric_answer(c.text, scorer) is None for c in all_candidates),
                  "scorer_version": scorer,
                  "split_counts": {s: sum(q.split == s for q in questions) for s in {q.split for q in questions}},
                  "is_benchmark_result": False,
                  "note": "Offline pool analysis; provenance and frozen held-out protocol must be reviewed before benchmark claims."})
    if selected is not None:
        stats["jev_vs_majority"] = paired_summary(outcomes["jev"], outcomes["majority_vote"], seed)
        stats["jev_vs_first_sample"] = paired_summary(outcomes["jev"], outcomes["first_sample"], seed)
        stats["gap_closed_vs_expected_random"] = gap_closed(stats["accuracy"]["jev"],
            stats["mean_single_sample_accuracy"], stats["empirical_oracle_at_pool_size"])
    run_dir = new_run(output, "pool-evaluation")
    if protocol is not None:
        stats["protocol"] = protocol
    write_json(run_dir / "metrics.json", stats)
    (run_dir / "evaluation_private").mkdir()
    write_json(run_dir / "evaluation_private/question_results.json", details)
    write_json(run_dir / "manifest.json", {"mode": "offline_evaluation", "master_seed": seed, "scorer_version": scorer,
        "protocol": protocol,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [questions_path, candidates_path, references_path] + ([selections_path] if selections_path else [])}})
    return run_dir
