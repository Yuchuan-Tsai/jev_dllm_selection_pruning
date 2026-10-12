"""Offline comparison of completed, matched development runs; never contacts Jev."""
import hashlib
import json
import math
from pathlib import Path
from .selection import evaluate_pools, load_pools, new_run, write_json


def comparable_runs(official: Path, fast: Path, questions: Path):
    runs = []
    expected_questions = hashlib.sha256(questions.read_bytes()).hexdigest()
    for path, expected_backend in ((official, "official_llada"), (fast, "fast_dllm_v1")):
        manifest = json.loads((path / "manifest.json").read_text())
        config = json.loads((path / "config.snapshot.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("questions_sha256") != expected_questions:
            raise ValueError("Backend comparison requires completed runs on the exact same question file")
        if config["generator"]["selected_backend"] != expected_backend:
            raise ValueError("Incorrect baseline/comparison backend")
        public_questions, pools = load_pools(questions, path / "candidates.jsonl")
        if any(q.dataset != "gsm8k" or q.split != "dev" for q in public_questions):
            raise ValueError("Backend comparison accepts GSM8K development inputs only")
        candidates = [c for pool in pools.values() for c in pool]
        if (manifest["completed_candidates"] != len(candidates)
                or any(len(pool) != config["generator"]["candidates_per_question"] for pool in pools.values())):
            raise ValueError("Run counts do not match complete candidate pools")
        costs = [json.loads(line) for line in (path / "costs.jsonl").read_text().splitlines() if line.strip()]
        by_id = {row["candidate_id"]: row for row in costs}
        if len(costs) != len(by_id) or set(by_id) != {c.candidate_id for c in candidates}:
            raise ValueError("Cost rows must cover every candidate exactly once")
        for candidate in candidates:
            cost = by_id[candidate.candidate_id]
            if cost["question_id"] != candidate.question_id:
                raise ValueError("Cost row belongs to another question")
            for field in ("trajectory_nfe", "generation_gpu_seconds", "generation_wall_seconds", "peak_memory_bytes"):
                value = cost[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid cost: {field}")
        runs.append((manifest, config, candidates, costs))
    a, b = runs
    settings = ("model_id", "model_revision", "tokenizer_revision", "dtype", "generation_length",
                "block_size", "temperature", "candidates_per_question", "microbatch_size")
    if any(a[1]["generator"][field] != b[1]["generator"][field] for field in settings):
        raise ValueError("Backend generation settings differ")
    if (a[1]["prompts"] != b[1]["prompts"]
            or a[1]["runtime"]["master_seed"] != b[1]["runtime"]["master_seed"]):
        raise ValueError("Backend prompts or master seeds differ")
    if [(c.question_id, c.candidate_id, c.seed) for c in a[2]] != [(c.question_id, c.candidate_id, c.seed) for c in b[2]]:
        raise ValueError("Backend candidate IDs, order or seeds differ")
    if (official / "dependencies.txt").read_bytes() != (fast / "dependencies.txt").read_bytes():
        raise ValueError("Backend dependency captures differ; review before comparing")
    return runs


def compare(official: Path, fast: Path, questions: Path, references: Path, output: Path,
            scorer="gsm8k_units_v2") -> Path:
    runs = comparable_runs(official, fast, questions)
    evaluations = [evaluate_pools(questions, path / "candidates.jsonl", references, output,
                                 scorer=scorer) for path in (official, fast)]
    summary = {"status": "complete", "mode": "development_backend_comparison", "scorer_version": scorer,
               "api_calls": 0, "runs": {}, "limitations": [
                   "Runs occurred in separate sessions; GPU load and clocks may differ.",
                   "Equal seeds do not imply equal trajectories across different samplers.",
                   "Caching and confidence-threshold decoding change compute and may change quality.",
                   "Timing ratios are descriptive, not a matched-budget Jev or pruning result."]}
    for (path, evaluation, values) in zip((official, fast), evaluations, runs):
        manifest, config, candidates, costs = values
        summary["runs"][config["generator"]["selected_backend"]] = {
            "input_run": str(path), "evaluation_run": str(evaluation),
            "metrics": json.loads((evaluation / "metrics.json").read_text()),
            "gpu": manifest.get("gpu"), "candidates": len(candidates),
            "nfe_total": sum(c["trajectory_nfe"] for c in costs),
            "gpu_seconds_total": sum(c["generation_gpu_seconds"] for c in costs),
            "candidate_wall_seconds_total": sum(c["generation_wall_seconds"] for c in costs),
            "peak_allocated_bytes": max(c["peak_memory_bytes"] for c in costs),
            "run_wall_seconds_including_load_and_warmup": manifest["total_seconds_including_load_and_warmup"]}
    a, b = [summary["runs"][name] for name in ("official_llada", "fast_dllm_v1")]
    summary["descriptive_gpu_time_ratio_official_over_fast"] = a["gpu_seconds_total"] / b["gpu_seconds_total"] if b["gpu_seconds_total"] > 0 else None
    run = new_run(output, "backend-comparison")
    write_json(run / "manifest.json", {"status": "complete", "mode": summary["mode"],
        "scorer_version": scorer, "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in (official, fast) for p in (d / "candidates.jsonl", d / "costs.jsonl", d / "config.snapshot.json", d / "manifest.json", d / "dependencies.txt")}})
    write_json(run / "comparison.json", summary)
    return run
