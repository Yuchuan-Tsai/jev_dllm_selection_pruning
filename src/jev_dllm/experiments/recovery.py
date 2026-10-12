"""Explicit candidate-boundary recovery; original frozen files and runs stay intact."""
from dataclasses import asdict, fields
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from . import frozen
from .selection import load_questions, new_run, write_json
from ..operations.remote import run_snapshot, snapshot_digest, valid_run_id, verify_copy
from ..reproducibility import trajectory_seed
from ..schema import Candidate, CostRecord

KIND = "gsm8k500_manual_candidate_recovery_v1"
EXTENSION_FILES = (
    "Makefile.recovery", "scripts/recovery.py",
    "src/jev_dllm/experiments/recovery.py", "src/jev_dllm/operations/recovery.py",
)


def extension_hashes(root):
    return {name: frozen.sha256(root / name) for name in EXTENSION_FILES}


def ordered_work(questions, config):
    return [(q, f"{q.question_id}/c{i:02}", trajectory_seed(config["runtime"]["master_seed"], q.question_id, i))
            for q in questions for i in range(config["generator"]["candidates_per_question"])]


def jsonl(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete last line in {path.name}; no automatic salvage")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSONL in {path.name}; no automatic salvage") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every saved record must be a JSON object")
    return rows


def check_rows(run, work, manifest):
    candidates = jsonl(run / "candidates.jsonl")
    costs = jsonl(run / "costs.jsonl")
    count = manifest.get("completed_candidates")
    if type(count) is not int or not 0 < count <= len(work) or len(candidates) != len(costs) or len(candidates) != count:
        raise ValueError("Saved candidate/cost/manifest counts disagree; no automatic salvage")
    for index, (row, cost) in enumerate(zip(candidates, costs)):
        question, candidate_id, seed = work[index]
        try:
            candidate = Candidate(**row)
        except TypeError as exc:
            raise ValueError("Saved candidate schema is invalid") from exc
        if (candidate.question_id, candidate.candidate_id, candidate.seed) != (question.question_id, candidate_id, seed):
            raise ValueError("Saved records are not the exact expected ordered ID/seed prefix")
        if (type(candidate.seed) is not int or type(candidate.completed) is not bool
                or type(candidate.truncated) is not bool or not isinstance(candidate.text, str)
                or (candidate.confidence is not None and (type(candidate.confidence) not in (int, float)
                    or not math.isfinite(candidate.confidence)))):
            raise ValueError("Invalid candidate fields")
        check_cost(cost, question.question_id, candidate_id)
    return count


def check_cost(cost, question_id, candidate_id):
    allowed = {f.name for f in fields(CostRecord)} | {"question_id", "candidate_id"}
    if set(cost) != allowed or (cost.get("question_id"), cost.get("candidate_id")) != (question_id, candidate_id):
        raise ValueError("Candidate and cost identities/schema differ")
    for name in allowed - {"question_id", "candidate_id"}:
        value = cost[name]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid recorded cost")
    if type(cost["trajectory_nfe"]) is not int or not 1 <= cost["trajectory_nfe"] <= 256:
        raise ValueError("Missing/invalid forward-call count")
    for name in ("generation_gpu_seconds", "generation_wall_seconds", "peak_memory_bytes"):
        if cost[name] is None:
            raise ValueError("Missing measured generation cost")


def receipt_for(run):
    valid_run_id(run.name)
    receipt = json.loads((run.parent / ".receipts" / (run.name + ".json")).read_text())
    snapshot = receipt["snapshot"]
    if snapshot.get("run_id") != run.name or receipt.get("digest") != snapshot_digest(snapshot):
        raise ValueError("Transfer receipt does not match this run")
    verify_copy(run.parent, snapshot)
    return snapshot


def inspect_run(root, run, *, snapshot=None):
    """Verify all saved records structurally without reading labels or scoring text."""
    snapshot = snapshot if snapshot is not None else receipt_for(run)
    if snapshot.get("run_id") != run.name:
        raise ValueError("Snapshot belongs to a different source run")
    verify_copy(run.parent, snapshot)
    config = json.loads((run / "config.snapshot.json").read_text())
    questions_path = root / config["datasets"]["questions_path"]
    protocol = frozen.check_protocol(root, config, questions_path)
    questions = load_questions(questions_path)
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest.get("protocol") != protocol or manifest.get("mode") != frozen.PHASE
            or manifest.get("question_count") != len(questions)
            or manifest.get("questions_sha256") != config["datasets"]["questions_sha256"]
            or manifest.get("backend") != "fast_dllm_v1"):
        raise ValueError("Source manifest does not match the frozen GSM8K generation protocol")
    work = ordered_work(questions, config)
    count = check_rows(run, work, manifest)
    if manifest.get("recovery"):
        check_lineage(run, manifest)
    return {"snapshot": snapshot, "config": config, "manifest": manifest,
            "protocol": protocol, "questions": questions, "work": work, "count": count}


def check_lineage(run, manifest):
    """Check copied-prefix bytes and source provenance, including repeated recovery."""
    info = manifest["recovery"]
    plan_path = run / "recovery-plan.json"
    plan = json.loads(plan_path.read_text())
    if info.get("kind") != KIND or info.get("plan_sha256") != frozen.sha256(plan_path):
        raise ValueError("Recovery plan provenance differs")
    if plan.get("kind") != KIND or plan.get("source_digest") != snapshot_digest(plan["source_snapshot"]):
        raise ValueError("Recovery source snapshot differs")
    if (plan["source_snapshot"].get("run_id") != plan["source_run_id"]
            or info.get("source_run_id") != plan["source_run_id"]
            or info.get("preserved_candidates") != plan["preserved_candidates"]
            or manifest["completed_candidates"] < plan["preserved_candidates"]
            or info.get("extension_sha256") != plan["extension_sha256"]
            or manifest.get("protocol") != plan["protocol"]):
        raise ValueError("Recovery lineage/count/implementation differs")
    for name in ("candidates.jsonl", "costs.jsonl"):
        expected = plan["source_snapshot"]["files"][name]
        with (run / name).open("rb") as stream:
            prefix = stream.read(expected["bytes"])
        import hashlib
        if len(prefix) != expected["bytes"] or hashlib.sha256(prefix).hexdigest() != expected["sha256"]:
            raise ValueError("Preserved source prefix changed")
    source_manifest = run / "source-manifest.json"
    if frozen.sha256(source_manifest) != plan["source_snapshot"]["files"]["manifest.json"]["sha256"]:
        raise ValueError("Original source manifest was not preserved")
    parent = json.loads(source_manifest.read_text())
    if (parent.get("completed_candidates") != plan["preserved_candidates"]
            or parent.get("protocol") != plan["protocol"]
            or parent.get("status") != "interrupted"
            or parent.get("error_type") != "KeyboardInterrupt"
            or manifest.get("prior_segments") != prior_segments(parent, plan["source_run_id"])):
        raise ValueError("Recovery segment timing history differs")


def prior_segments(manifest, run_id):
    history = list(manifest.get("prior_segments", []))
    history.append({"run_id": run_id, "status": manifest["status"],
        "reused_candidates": manifest.get("recovery", {}).get("preserved_candidates", 0),
        "ending_candidate_count": manifest["completed_candidates"],
        "segment_seconds_including_load_and_warmup": manifest.get("total_seconds_including_load_and_warmup"),
        "note": "Includes interrupted work; per-candidate costs omit an unfinished candidate."})
    return history


def summary(info):
    count, work = info["count"], info["work"]
    return {"status": info["manifest"]["status"], "source_run_id": info["snapshot"]["run_id"],
        "verified_source_digest": snapshot_digest(info["snapshot"]), "preserved_candidates": count,
        "remaining_candidates": len(work) - count,
        "next_candidate_id": work[count][1] if count < len(work) else None,
        "next_seed": work[count][2] if count < len(work) else None,
        "correctness_scored": False, "gpu_or_api_work_started": False}


def prepare_plan(root, run, destination=None):
    info = inspect_run(root, run)
    m = info["manifest"]
    if m.get("status") != "interrupted" or m.get("error_type") != "KeyboardInterrupt" or info["count"] >= len(info["work"]):
        raise ValueError("Manual recovery requires an interrupted, incomplete run; do not retry failed/complete runs blindly")
    plan = {"schema_version": 1, "kind": KIND, "source_run_id": run.name,
        "source_snapshot": info["snapshot"], "source_digest": snapshot_digest(info["snapshot"]),
        "protocol": info["protocol"], "extension_sha256": extension_hashes(root),
        "preserved_candidates": info["count"], "total_candidates": len(info["work"]),
        "next_candidate_id": info["work"][info["count"]][1],
        "next_seed": info["work"][info["count"]][2],
        "policy": "User-requested operational recovery only; preserve every saved record, regenerate missing candidates with original seeds, never select by quality.",
        "limitation": "Candidate boundary recovery, not restoration of an interrupted diffusion trajectory; GPU bitwise equivalence is not asserted."}
    destination = destination or root / "artifacts/recovery-plans" / (run.name + ".json")
    if destination.exists():
        if json.loads(destination.read_text()) != plan:
            raise ValueError("Existing recovery plan differs; preserve it and use a new explicit destination")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return {**summary(info), "plan": str(destination), "plan_sha256": frozen.sha256(destination)}


def check_plan(root, plan_path):
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != 1 or plan.get("kind") != KIND:
        raise ValueError("Unknown recovery plan")
    valid_run_id(plan["source_run_id"])
    if plan.get("source_digest") != snapshot_digest(plan["source_snapshot"]):
        raise ValueError("Recovery snapshot digest differs")
    if plan.get("extension_sha256") != extension_hashes(root):
        raise ValueError("Recovery implementation differs from the audited plan")
    return plan


def check_server_source(root, plan_path):
    plan = check_plan(root, plan_path)
    run = root / "artifacts/runs" / plan["source_run_id"]
    current = run_snapshot(root, run.name)
    if snapshot_digest(current) != plan["source_digest"]:
        raise ValueError("Server source differs from the verified local copy")
    info = inspect_run(root, run, snapshot=current)
    if (info["manifest"].get("status") != "interrupted"
            or info["manifest"].get("error_type") != "KeyboardInterrupt"
            or info["count"] != plan["preserved_candidates"]
            or len(info["work"]) != plan["total_candidates"] or info["protocol"] != plan["protocol"]
            or not info["count"] < len(info["work"])
            or info["work"][info["count"]][1:] != (plan["next_candidate_id"], plan["next_seed"])):
        raise ValueError("Recovery plan does not match the interrupted source")
    return plan, run, info


def runtime_check(root, config):
    from ..operations.dependencies import check_lock
    from ..operations.server import gpu_check
    import torch
    check_lock(root / "requirements/gpu-linux-py311.lock")
    hardware = gpu_check(root)
    frozen.check_hardware(config, torch)
    return hardware


def load_backend(root, config):
    from ..backends.fast_llada import FastLLaDA
    return FastLLaDA(config, root, (root / config["prompts"]["generation_math"]["path"]).read_text(), False)


def recover(root, plan_path):
    plan, source, info = check_server_source(root, plan_path)
    config, work, preserved = info["config"], info["work"], info["count"]
    hardware = runtime_check(root, config)
    dependencies = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    if dependencies != info["manifest"].get("dependencies"):
        raise ValueError("Runtime versions differ from the original source run")
    # One explicit attempt per immutable source: reject concurrent/accidental duplicate launches.
    claims = root / "artifacts/recovery-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim = claims / (source.name + ".json")
    try:
        claim_stream = claim.open("x")
    except FileExistsError as exc:
        raise ValueError(f"Recovery was already attempted for this source. Inspect {claim}; after another interruption, archive the NEW run and prepare its own plan.") from exc
    with claim_stream:
        run = new_run(root / "artifacts/runs", "fast-llada-gsm8k500-recovery")
        json.dump({"run_id": run.name, "plan_sha256": frozen.sha256(plan_path)}, claim_stream)
        claim_stream.write("\n")
    print(f"Recovery run directory: {run}", flush=True)
    print(f"Preserving {preserved}/{len(work)} candidates; generating {len(work)-preserved} missing candidates", flush=True)
    m = {"status": "initializing", "mode": frozen.PHASE, "protocol": info["protocol"],
        "is_benchmark_result": False, "question_count": len(info["questions"]),
        "completed_candidates": preserved, "pid": os.getpid(), "backend": "fast_dllm_v1",
        "questions_sha256": config["datasets"]["questions_sha256"],
        "python": platform.python_version(), "platform": platform.platform(), "dependencies": dependencies,
        "gpu": hardware["gpu_name"], "hardware_at_start": hardware,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
        "warmup": "one additional unscored first-question call for this recovery segment",
        "prior_segments": prior_segments(info["manifest"], source.name),
        "recovery": {"kind": KIND, "source_run_id": source.name, "preserved_candidates": preserved,
            "plan_sha256": frozen.sha256(plan_path), "extension_sha256": plan["extension_sha256"],
            "in_flight_candidate": "Restart from its original seed; no saved token/cache state.",
            "timing_scope": "costs.jsonl covers all completed candidates; elapsed fields in this manifest cover ONLY this recovery segment; add prior_segments separately, excluding offline downtime.",
            "unrecorded_partial_candidate_gpu_seconds": None}}
    write_json(run / "manifest.json", m)
    started = time.perf_counter()
    generation_start = None
    try:
        for name in ("config.snapshot.json", "candidates.jsonl", "costs.jsonl"):
            shutil.copyfile(source / name, run / name)
        shutil.copyfile(source / "manifest.json", run / "source-manifest.json")
        shutil.copyfile(plan_path, run / "recovery-plan.json")
        check_rows(run, work, m)
        check_lineage(run, m)
        deployment = root / "artifacts/deployment.json"
        if deployment.exists():
            shutil.copyfile(deployment, run / "deployment.json")
            m["deployment_sha256"] = frozen.sha256(deployment)
            m["project_commit"] = json.loads(deployment.read_text()).get("project_commit")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (run / "dependencies.txt").write_text(freeze.stdout)
        m["dependency_capture_succeeded"] = True
        load_start = time.perf_counter()
        backend = load_backend(root, config)
        m["model_load_seconds"] = time.perf_counter() - load_start
        warm_start = time.perf_counter()
        backend.generate(info["questions"][0], candidate_id="warmup", seed=config["runtime"]["master_seed"])
        m["warmup_seconds"] = time.perf_counter() - warm_start
        generation_start = time.perf_counter()
        m["status"] = "running"
        write_json(run / "manifest.json", m)
        with (run / "candidates.jsonl").open("a") as candidates, (run / "costs.jsonl").open("a") as costs:
            for question, candidate_id, seed in work[preserved:]:
                candidate, cost = backend.generate(question, candidate_id=candidate_id, seed=seed)
                if (candidate.question_id, candidate.candidate_id, candidate.seed) != (question.question_id, candidate_id, seed):
                    raise ValueError("Backend returned an unexpected candidate identity")
                cost_row = {"question_id": question.question_id, "candidate_id": candidate_id, **asdict(cost)}
                check_cost(cost_row, question.question_id, candidate_id)
                candidates.write(json.dumps(asdict(candidate), allow_nan=False) + "\n")
                costs.write(json.dumps(cost_row, allow_nan=False) + "\n")
                candidates.flush()
                costs.flush()
                m["completed_candidates"] += 1
                write_json(run / "manifest.json", m)
                print(f"Completed {m['completed_candidates']}/{len(work)} candidates (preserved {preserved})", flush=True)
        check_rows(run, work, m)
        check_lineage(run, m)
        if snapshot_digest(run_snapshot(root, source.name)) != plan["source_digest"]:
            raise ValueError("Original source changed during recovery")
        m["status"] = "complete"
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        m["total_seconds_including_load_and_warmup"] = time.perf_counter() - started
        if generation_start is not None:
            m["generation_phase_seconds"] = time.perf_counter() - generation_start
        m["newly_generated_candidates"] = m["completed_candidates"] - preserved
        write_json(run / "manifest.json", m)
    return run
