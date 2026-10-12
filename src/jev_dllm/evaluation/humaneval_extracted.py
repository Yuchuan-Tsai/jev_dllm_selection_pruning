"""Explicit HumanEval extraction reanalysis; preserve strict evidence and raw pools."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from . import humaneval_isolated as strict
from .code_extraction import extract_python, fixtures, VERSION as EXTRACTOR_VERSION
from .metrics import pool_metrics
from ..experiments import collection, frozen
from ..experiments.selection import load_pools, new_run, write_json
from ..reproducibility import trajectory_seed
from ..selection.baselines import random_choice

PROFILE = "configs/evaluation/humaneval-extracted-v2.json"
VERSION = "humaneval_single_block_v2"
FILES = (PROFILE, "src/jev_dllm/evaluation/code_extraction.py",
         "src/jev_dllm/evaluation/humaneval_extracted.py", "scripts/reanalyze_humaneval.py") + strict.FILES


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_policy(root):
    policy = json.loads((root / PROFILE).read_text())
    if policy["version"] != VERSION or policy["extractor"] != EXTRACTOR_VERSION or policy["analysis_type"] != "post_hoc_reanalysis":
        raise ValueError("Unsupported extraction policy")
    if set(policy["strict_implementation_sha256"]) != set(strict.FILES):
        raise ValueError("Incomplete strict-runtime pins")
    for p, h in policy["strict_implementation_sha256"].items():
        if frozen.sha256(root / p) != h:
            raise ValueError(f"Original strict implementation changed: {p}")
    return policy


def extraction_record(candidate):
    e = extract_python(candidate.text)
    return e, {"question_id": candidate.question_id, "candidate_id": candidate.candidate_id,
               "raw_text_sha256": sha_text(candidate.text), "extraction_mode": e.mode,
               "extraction_reason": e.reason, "source_span": [e.start, e.end] if e.source is not None else None,
               "extracted_text_sha256": sha_text(e.source) if e.source is not None else None}


def score_candidate(profile, worker, candidate, reference):
    e, record = extraction_record(candidate)
    if not candidate.completed or candidate.truncated or not candidate.text.strip():
        return {**record, "status": "generation_flag_failure", "seconds": 0.0}
    if e.source is None:
        return {**record, "status": "extraction_failure", "seconds": 0.0}
    if e.source != candidate.text[e.start:e.end]:
        raise ValueError("Extraction must preserve an exact source substring")
    return {**record, **strict.score_one(profile, worker, e.source, reference)}


def validate_extraction(profile, worker):
    rows = []
    reference = {"entry_point": "increment", "hidden_tests": "def check(candidate):\n    assert candidate(2) == 3\n    assert candidate(-4) == -3\n"}
    for name, raw, mode, source, expected in fixtures():
        e = extract_python(raw)
        if e.mode != mode or e.source != source:
            raise strict.InfrastructureError(f"Extraction fixture failed: {name}")
        if e.source is not None:
            if e.source != raw[e.start:e.end]:
                raise strict.InfrastructureError("Fixture source was rewritten")
            result = strict.score_one(profile, worker, e.source, reference)
        else:
            result = {"status": "extraction_failure"}
        if result["status"] != expected:
            raise strict.InfrastructureError(f"Extraction execution fixture failed: {name}")
        rows.append({"fixture": name, "mode": e.mode, "expected": expected, **result})
    return {"status": "passed", "fixtures": rows, "source": "independent handwritten examples"}


def verify_strict(root, run, strict_run):
    manifest = json.loads((strict_run / "manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["profile"]["version"] != "humaneval_full_source_strict_v1"
            or Path(manifest["source_run"]).resolve() != run.resolve()
            or manifest["results_sha256"] != frozen.sha256(strict_run / "candidate-results.jsonl")):
        raise ValueError("Require the completed strict evaluation of this exact archive")
    if set(manifest["implementation_sha256"]) != set(strict.FILES):
        raise ValueError("Strict implementation provenance is incomplete")
    for p, h in manifest["implementation_sha256"].items():
        if frozen.sha256(root / p) != h:
            raise ValueError("Strict scorer has changed")
    for p, h in manifest["input_sha256"].items():
        if frozen.sha256(Path(p)) != h:
            raise ValueError("Strict evaluation inputs changed")
    return manifest


def summarize(questions, pools, results, seed):
    expected = [(q.question_id, c.candidate_id) for q in questions for c in pools[q.question_id]]
    if [(r["question_id"], r["candidate_id"]) for r in results] != expected:
        raise ValueError("Results must cover the original ordered candidates exactly once")
    if any(r["status"] not in strict.STATUSES | {"extraction_failure", "generation_flag_failure"} for r in results):
        raise ValueError("Infrastructure/unknown outcomes cannot be scored incorrect")
    lookup = {r["candidate_id"]: r["status"] == "passed" for r in results}
    labels = {q.question_id: [lookup[c.candidate_id] for c in pools[q.question_id]] for q in questions}
    metrics = pool_metrics({q.question_id: [c.text for c in pools[q.question_id]] for q in questions}, labels)
    metrics.pop("candidate_counts")
    first = sum(v[0] for v in labels.values())
    random_count = sum(lookup[random_choice(pools[q.question_id], trajectory_seed(seed, q.question_id, 1))] for q in questions)
    metrics.update({"candidate_count": len(results), "passed_candidates": sum(lookup.values()),
                    "first_sample": {"correct": first, "total": len(questions), "accuracy": first/len(questions)},
                    "seeded_random": {"correct": random_count, "total": len(questions), "accuracy": random_count/len(questions)},
                    "solvable_pools": sum(any(v) for v in labels.values()),
                    "all_correct_pools": sum(all(v) for v in labels.values()),
                    "pass_count_histogram": dict(sorted(Counter(sum(v) for v in labels.values()).items())),
                    "status_counts": dict(Counter(r["status"] for r in results)),
                    "extraction_mode_counts": dict(Counter(r["extraction_mode"] for r in results)),
                    "extraction_rejection_counts": dict(Counter(r["extraction_reason"] for r in results if r["extraction_reason"])),
                    "scorer_version": VERSION, "analysis_type": "post_hoc_reanalysis", "jev_evaluated": False,
                    "is_benchmark_result": False, "note": "Same saved pools; only versioned source extraction changed. Original strict results retained. No new model generation or Jev requests."})
    return metrics


def compare_strict(old, new):
    if [(r["question_id"], r["candidate_id"]) for r in old] != [(r["question_id"], r["candidate_id"]) for r in new]:
        raise ValueError("Strict/extracted result alignment differs")
    return {"candidate_transitions": dict(Counter(f'{a["status"]} -> {b["status"]}' for a,b in zip(old,new))),
            "pass_gains": sum(a["status"] != "passed" and b["status"] == "passed" for a,b in zip(old,new)),
            "pass_losses": sum(a["status"] == "passed" and b["status"] != "passed" for a,b in zip(old,new)),
            "unchanged_source_status_changes": [b["candidate_id"] for a,b in zip(old,new)
                                                 if b["extraction_mode"] == "raw" and a["status"] != b["status"]]}


def evaluate(root, run=None, strict_run=None, validate_only=False):
    policy = load_policy(root)
    profile = strict.load_profile(root)
    worker = (root / strict.WORKER).read_text()
    hashes = {p: frozen.sha256(root / p) for p in FILES}
    out = new_run(root / "artifacts/runs", "humaneval-extraction-validation" if validate_only else "humaneval-extracted-evaluation")
    print(f"Evaluation directory: {out}", flush=True)
    m = {"status": "validating", "started_at_utc": datetime.now(timezone.utc).isoformat(),
         "policy": policy, "runtime_profile": profile, "implementation_sha256": hashes,
         "host_python_for_static_extraction": platform.python_version(), "source_run": str(run) if run else None,
         "strict_evaluation": str(strict_run) if strict_run else None, "completed_candidates": 0,
         "jev_requests": 0, "new_generation": False, "analysis_type": "post_hoc_reanalysis"}
    write_json(out / "manifest.json", m)
    started = time.monotonic()
    try:
        write_json(out / "runtime-validation.json", strict.validate_runtime(root, profile, worker))
        write_json(out / "extraction-validation.json", validate_extraction(profile, worker))
        if validate_only:
            m["status"] = "complete"
            write_json(out / "manifest.json", m)
            return out
        if run is None or strict_run is None:
            raise ValueError("Provide the archive and its original strict evaluation")
        audit = collection.audit(root, run)
        if audit["status"] != "complete" or audit["task"] != "humaneval":
            raise ValueError("Require a complete verified HumanEval archive")
        write_json(out / "collection-audit.json", audit)
        old_manifest = verify_strict(root, run, strict_run)
        m["input_sha256"] = dict(old_manifest["input_sha256"])
        m["strict_evidence_sha256"] = {str(p): frozen.sha256(p) for p in sorted(strict_run.rglob("*")) if p.is_file()}
        write_json(out / "manifest.json", m)
        qp = root / "data/processed/humaneval-heldout.json"
        rp = root / "data/private/humaneval-heldout-references.json"
        data = json.loads((root / "data/manifests/humaneval-week01.json").read_text())
        if frozen.sha256(rp) != data["files"][str(rp.relative_to(root))]:
            raise ValueError("Private reference hash mismatch")
        questions, pools = load_pools(qp, run / "candidates.jsonl")
        references = json.loads(rp.read_text())
        refs = {r["question_id"]: r for r in references}
        if len(refs) != len(references) or set(refs) != set(pools):
            raise ValueError("References must cover the exact public task set")
        controls = strict.reference_controls(profile, worker, questions, refs)
        write_json(out / "reference-controls.json", controls)
        if any(r["status"] != "passed" for r in controls):
            raise strict.InfrastructureError("Official reference control failed")
        print(f"Official reference controls passed: {len(controls)}/{len(controls)}", flush=True)
        m["status"] = "scoring"
        write_json(out / "manifest.json", m)
        candidates = [c for q in questions for c in pools[q.question_id]]
        rows = []
        with (out / "candidate-results.jsonl").open("w") as stream, ThreadPoolExecutor(max_workers=profile["workers"]) as executor:
            for offset in range(0, len(candidates), profile["workers"]):
                jobs = [executor.submit(score_candidate, profile, worker, c, refs[c.question_id])
                        for c in candidates[offset:offset + profile["workers"]]]
                for job in jobs:
                    result = job.result()
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    rows.append(result)
                    m["completed_candidates"] = len(rows)
                write_json(out / "manifest.json", m)
                if len(rows) % 64 == 0 or len(rows) == len(candidates):
                    print(f"HumanEval extraction reanalysis: {len(rows)}/{len(candidates)}", flush=True)
        for field in ("input_sha256", "strict_evidence_sha256"):
            if any(frozen.sha256(Path(p)) != h for p,h in m[field].items()):
                raise ValueError("An original scoring input or strict artifact changed")
        if hashes != {p: frozen.sha256(root / p) for p in FILES}:
            raise ValueError("Scoring implementation changed during analysis")
        old = [json.loads(s) for s in (strict_run / "candidate-results.jsonl").read_text().splitlines()]
        write_json(out / "strict-comparison.json", compare_strict(old, rows))
        metrics = summarize(questions, pools, rows, profile["master_seed"])
        metrics["scoring_wall_seconds_including_validation"] = time.monotonic() - started
        write_json(out / "metrics.json", metrics)
        m.update(status="complete", results_sha256=frozen.sha256(out / "candidate-results.jsonl"),
                 finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", m)
        return out
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        write_json(out / "manifest.json", m)
        raise
