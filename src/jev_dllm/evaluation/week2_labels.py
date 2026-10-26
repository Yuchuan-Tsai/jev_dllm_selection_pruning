"""Private correctness labels and pool statistics for verified Week 2 development pools.

Labels reuse each task's Week 1 scorer: GSM8K numeric units v3, the frozen MATH500
Math-Verify policy, and HumanEval single-block extraction in offline Docker for MBPP.
Labels stay in evaluation_private/; metrics.json holds only aggregate statistics.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

from . import humaneval_extracted, humaneval_isolated as strict, math500
from .answers import numeric_answer, numeric_value
from .metrics import pool_metrics
from ..data import week2_dev
from ..experiments import week2_pool
from ..experiments.selection import new_run, write_json
from ..operations import remote
from ..schema import Candidate
from ..selection.subsets import SIZES, nested_subsets

SCORERS = {"gsm8k": "gsm8k_units_v3", "math": "math500_math_verify_v1", "mbpp": "humaneval_single_block_v2"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_pool(root: Path, run: Path):
    """Accept only a complete, SHA256-verified local copy of an unchanged-profile pool run."""
    remote.valid_run_id(run.name)
    receipt = json.loads((run.parent / ".receipts" / (run.name + ".json")).read_text())
    if receipt["snapshot"].get("run_id") != run.name or receipt["digest"] != remote.snapshot_digest(receipt["snapshot"]):
        raise ValueError("Invalid archive receipt; run results-verify first")
    remote.verify_copy(run.parent, receipt["snapshot"])
    manifest = json.loads((run / "manifest.json").read_text())
    task = manifest.get("task")
    profile, base, questions = week2_pool.load_profile(root, task)
    if (manifest.get("mode") != week2_pool.SCOPE or manifest.get("status") != "complete"
            or (run / "pool-profile.json").read_bytes() != (root / week2_pool.profile_path(task)).read_bytes()):
        raise ValueError("Require a complete Week 2 pool run made with the current pool profile")
    rows = {name: week2_pool.read_rows(run / name) for name in week2_pool.FILES}
    schedule = week2_pool.plan(questions, profile["candidates_per_question"], base["runtime"]["master_seed"])
    if any(len(r) != len(schedule) for r in rows.values()):
        raise ValueError("Pool run is missing candidate, cost or trajectory records")
    pools = {q.question_id: [] for q in questions}
    for (question, cid, seed), c, cost, trace in zip(schedule, *rows.values()):
        if (c["candidate_id"], c["seed"], cost["candidate_id"], trace["candidate_id"]) != (cid, seed, cid, cid):
            raise ValueError("Pool records do not follow the planned order and seeds")
        pools[question.question_id].append(Candidate(**c))
    return task, manifest, questions, pools, rows["costs.jsonl"], rows["trajectories.jsonl"]


def references(root: Path, task: str, questions) -> dict:
    relative = f"data/private/{task}-w2dev-references.json"
    if sha256(root / relative) != json.loads((root / week2_dev.MANIFEST).read_text())["files"][relative]:
        raise ValueError("Private references differ from the development manifest")
    refs = {r["question_id"]: r for r in json.loads((root / relative).read_text())}
    if set(refs) != {q.question_id for q in questions}:
        raise ValueError("References must cover exactly the pool questions")
    return refs


def gsm8k_labels(questions, pools, refs):
    rows = []
    for q in questions:
        gold = numeric_value(refs[q.question_id]["answer"])
        for c in pools[q.question_id]:
            valid = c.completed and not c.truncated
            value = numeric_answer(c.text, SCORERS["gsm8k"]) if valid else None
            rows.append({"candidate_id": c.candidate_id, "correct": bool(value is not None and value == gold),
                         "status": "invalid_candidate" if not valid else "unparseable" if value is None else "parsed",
                         "extracted": None if value is None else str(value)})
    return rows, {}


def math_labels(root, python, questions, pools, refs, out):
    policy = math500.read(root / math500.POLICY)
    validation = math500.worker(root, python, "validate", policy, controls=math500.read(root / math500.CONTROLS))
    control = math500.worker(root, python, "reference-controls", policy,
                             references=[{"question_id": q.question_id, "answer": refs[q.question_id]["answer"]} for q in questions])
    write_json(out / "evaluation_private/math-validation.json", validation)
    write_json(out / "evaluation_private/reference-controls.json", control)
    if validation["status"] != "passed" or control["status"] != "passed":
        raise ValueError("Math scorer fixtures or reference controls failed; inspect before labeling")

    def grade(q):
        candidates = [{k: getattr(c, k) for k in ("candidate_id", "text", "completed", "truncated")} for c in pools[q.question_id]]
        return math500.worker(root, python, "question", policy, question={
            "question_id": q.question_id, "candidates": candidates, "reference": refs[q.question_id]["answer"]})

    rows = []
    with ThreadPoolExecutor(max_workers=policy["workers"]) as executor:
        for record in executor.map(grade, questions):
            rows += [{"candidate_id": r["candidate_id"], "correct": r["correct"], "status": r["extraction_status"],
                      "extracted": r["parsed"], "verification": r["verification_status"]} for r in record["candidates"]]
    return rows, {"runtime": validation["runtime"]}


def mbpp_labels(root, questions, pools, refs, out):
    profile, policy = strict.load_profile(root), humaneval_extracted.load_policy(root)
    worker = (root / strict.WORKER).read_text()
    runtime = strict.validate_runtime(root, profile, worker)
    extraction = humaneval_extracted.validate_extraction(profile, worker)
    controls = strict.reference_controls(profile, worker, questions, refs)
    write_json(out / "evaluation_private/runtime-validation.json", {"runtime": runtime, "extraction": extraction})
    write_json(out / "evaluation_private/reference-controls.json", controls)
    # A reference that fails its own converted tests marks the task unusable, independent of model outputs.
    excluded = sorted(r["question_id"] for r in controls if r["status"] != "passed")
    usable = [c for q in questions if q.question_id not in excluded for c in pools[q.question_id]]
    rows = []
    with ThreadPoolExecutor(max_workers=profile["workers"]) as executor:
        for offset in range(0, len(usable), profile["workers"]):
            batch = usable[offset:offset + profile["workers"]]
            for c, result in zip(batch, executor.map(lambda c: humaneval_extracted.score_candidate(
                    profile, worker, c, refs[c.question_id]), batch)):
                rows.append({"candidate_id": c.candidate_id, "correct": result["status"] == "passed",
                             "status": result["status"], "extraction_mode": result["extraction_mode"]})
    return rows, {"excluded_questions_failing_reference_tests": excluded, "docker": runtime["runtime"],
                  "policy_version": policy["version"]}


def statistics(questions, pools, correct, costs, traces, run_note=None) -> dict:
    """Aggregate pool quality per nested N; questions without labels are excluded and counted."""
    labeled = [q for q in questions if all(c.candidate_id in correct for c in pools[q.question_id])]
    sizes = tuple(n for n in SIZES if n <= min(len(pools[q.question_id]) for q in labeled))
    by_n = {}
    for n in sizes:
        texts, labels, first = {}, {}, []
        for q in labeled:
            pool = {c.candidate_id: c for c in pools[q.question_id]}
            ids = nested_subsets(list(pool), sizes)[n]
            texts[q.question_id] = [pool[i].text for i in ids]
            labels[q.question_id] = [correct[i] for i in ids]
            first.append(correct[ids[0]])
        stats = pool_metrics(texts, labels)
        stats.pop("candidate_counts")
        by_n[str(n)] = {**stats, "first_in_subset_accuracy": mean(first)}
    candidates = [c for q in labeled for c in pools[q.question_id]]
    cost = {c["candidate_id"]: c for c in costs}
    confidence = {t["candidate_id"]: t["confidence"]["mean_probability"] for t in traces}
    scored = [c.candidate_id for c in candidates]

    def conf(flag):
        values = [confidence[i] for i in scored if correct[i] is flag and confidence[i] is not None]
        return {"count": len(values), "mean": mean(values) if values else None}

    return {"question_count": len(labeled), "excluded_questions": len(questions) - len(labeled),
            "candidate_count": len(candidates), "by_subset_size": by_n,
            "generation_flags": {"truncated_fraction": mean(c.truncated for c in candidates),
                                 "incomplete_fraction": mean(not c.completed for c in candidates),
                                 "empty_fraction": mean(not c.text.strip() for c in candidates)},
            "generation_cost": {"mean_nfe": mean(cost[i]["trajectory_nfe"] for i in scored),
                                "total_nfe": sum(cost[i]["trajectory_nfe"] for i in scored),
                                "mean_gpu_seconds": mean(cost[i]["generation_gpu_seconds"] for i in scored),
                                "mean_wall_seconds": mean(cost[i]["generation_wall_seconds"] for i in scored),
                                "timing_note": run_note or "No run note recorded; timing assumes an unshared GPU."},
            "commit_confidence_by_correctness": {"correct": conf(True), "incorrect": conf(False)},
            "subset_rule": "first N candidates in pool order (same prefixes as week2_jev requests)",
            "unit_of_analysis": "question",
            "note": "Development pool statistics. Oracle uses private labels and is not a selector. "
                    "Commit confidence is descriptive here; selector comparisons belong to Week 3."}


def label(root: Path, run: Path, math_python: Path | None = None) -> Path:
    task, pool_manifest, questions, pools, costs, traces = audit_pool(root, run)
    refs = references(root, task, questions)
    out = new_run(root / "artifacts/runs", f"w2labels-{task}")
    (out / "evaluation_private").mkdir()
    m = {"status": "labeling", "task": task, "scorer": SCORERS[task], "pool_run": run.name,
         "pool_inputs_sha256": {name: sha256(run / name) for name in week2_pool.FILES},
         "references_sha256": sha256(root / f"data/private/{task}-w2dev-references.json"),
         "started_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0}
    write_json(out / "manifest.json", m)
    print(f"Labels directory: {out}", flush=True)
    started = time.monotonic()
    try:
        if task == "gsm8k":
            rows, details = gsm8k_labels(questions, pools, refs)
        elif task == "math":
            if math_python is None:
                raise ValueError("MATH labels need the isolated Math-Verify interpreter (make math500-scoring-setup)")
            rows, details = math_labels(root, math_python, questions, pools, refs, out)
        else:
            rows, details = mbpp_labels(root, questions, pools, refs, out)
        with (out / "evaluation_private/labels.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        correct = {r["candidate_id"]: r["correct"] for r in rows}
        metrics = statistics(questions, pools, correct, costs, traces, run_note=pool_manifest.get("run_note"))
        metrics.update(task=task, scorer=SCORERS[task], details=details)
        write_json(out / "metrics.json", metrics)
        if any(sha256(run / name) != digest for name, digest in m["pool_inputs_sha256"].items()):
            raise ValueError("Pool files changed during labeling")
        m.update(status="complete", elapsed_seconds=time.monotonic() - started,
                 labels_sha256=sha256(out / "evaluation_private/labels.jsonl"), metrics_sha256=sha256(out / "metrics.json"))
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        write_json(out / "manifest.json", m)
    return out
