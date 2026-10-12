"""Verified candidate-boundary continuation of pinned benchmark collection."""
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from . import collection, frozen
from .selection import new_run, write_json
from ..operations.remote import run_snapshot, snapshot_digest, valid_run_id, verify_copy
from ..reproducibility import trajectory_seed
from ..schema import Candidate, CostRecord

KIND = "benchmark_collection_candidate_recovery_v1"
EXTENSION_FILES = (
    "Makefile.benchmarks-recovery", "scripts/recover_collection.py",
    "src/jev_dllm/experiments/collection_recovery.py", "src/jev_dllm/operations/collection_recovery.py",
)


def extension_hashes(root):
    for name in EXTENSION_FILES:
        if any((root / part).is_symlink() for part in (Path(name), *Path(name).parents)):
            raise ValueError("Recovery implementation must not contain symlinks")
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
    candidates, costs = jsonl(run / "candidates.jsonl"), jsonl(run / "costs.jsonl")
    count = manifest.get("completed_candidates")
    if (type(count) is not int or not 0 < count <= len(work)
            or len(candidates) != count or len(costs) != count
            or (manifest.get("status") == "complete" and count != len(work))):
        raise ValueError("Saved candidate/cost/manifest counts disagree; no automatic salvage")
    flags = {"incomplete_candidates": 0, "truncated_candidates": 0, "empty_candidates": 0}
    cost_fields = {f.name for f in fields(CostRecord)}
    for index, (row, cost) in enumerate(zip(candidates, costs)):
        question, cid, seed = work[index]
        if (set(cost) != cost_fields | {"question_id", "candidate_id"}
                or (cost["question_id"], cost["candidate_id"]) != (question.question_id, cid)):
            raise ValueError("Candidate and cost identities/schema differ")
        if set(row) != {f.name for f in fields(Candidate)}:
            raise ValueError("Saved candidate schema differs")
        candidate = Candidate(**row)
        collection.validate_record(candidate, CostRecord(**{key: cost[key] for key in cost_fields}),
                                   question, cid, seed)
        flags["incomplete_candidates"] += int(not candidate.completed)
        flags["truncated_candidates"] += int(candidate.truncated)
        flags["empty_candidates"] += int(not candidate.text.strip())
    if any(type(manifest.get(key)) is not int or manifest[key] != value for key, value in flags.items()):
        raise ValueError("Saved completion flags disagree with the manifest")
    return count


def receipt_for(run):
    valid_run_id(run.name)
    receipt = json.loads((run.parent / ".receipts" / (run.name + ".json")).read_text())
    snapshot = receipt["snapshot"]
    if snapshot.get("run_id") != run.name or receipt.get("digest") != snapshot_digest(snapshot):
        raise ValueError("Transfer receipt does not match this run")
    verify_copy(run.parent, snapshot)
    return snapshot


def inspect_run(root, run, *, snapshot=None):
    """Validate the ordered public-only prefix, never score or filter answers."""
    snapshot = snapshot if snapshot is not None else receipt_for(run)
    if snapshot.get("run_id") != run.name:
        raise ValueError("Snapshot belongs to a different source run")
    verify_copy(run.parent, snapshot)
    manifest = json.loads((run / "manifest.json").read_text())
    task = manifest["task"]
    spec, questions = collection.check(root, task)
    config, protocol = spec["backend_profile"], collection.contract(root, task)
    if (manifest.get("mode") != collection.SCOPE or manifest.get("protocol") != protocol
            or manifest.get("status") not in ("interrupted", "failed", "complete")
            or snapshot.get("status") != manifest["status"]
            or manifest.get("question_count") != len(questions)
            or manifest.get("candidates_per_question") != 8
            or manifest.get("evaluation_ready") is not False
            or manifest.get("is_benchmark_result") is not False
            or manifest.get("api_calls") != 0
            or manifest.get("questions_sha256") != config["datasets"]["questions_sha256"]
            or json.loads((run / "collection-profile.json").read_text()) != spec
            or json.loads((run / "config.snapshot.json").read_text()) != config
            or frozen.sha256(run / "generation-lock.json") != frozen.sha256(root / collection.LOCK)):
        raise ValueError("Source manifest/snapshots do not match the pinned collection")
    work = ordered_work(questions, config)
    count = check_rows(run, work, manifest)
    if "recovery" in manifest:
        check_lineage(root, run, manifest, work)
        if manifest.get("newly_generated_candidates") != count - manifest["recovery"]["preserved_candidates"]:
            raise ValueError("Recovery segment candidate count differs")
    elif "prior_segments" in manifest or (run / "recovery-plan.json").exists():
        raise ValueError("Recovery metadata is incomplete")
    return {"snapshot": snapshot, "config": config, "manifest": manifest,
            "protocol": protocol, "questions": questions, "work": work, "count": count}


def check_lineage(root, run, manifest, work):
    """Check copied-prefix bytes and source provenance, including repeated recovery."""
    info = manifest["recovery"]
    plan_path = run / "recovery-plan.json"
    plan = check_plan(root, plan_path)
    if info.get("kind") != KIND or info.get("plan_sha256") != frozen.sha256(plan_path):
        raise ValueError("Recovery plan provenance differs")
    if plan.get("kind") != KIND or plan.get("source_digest") != snapshot_digest(plan["source_snapshot"]):
        raise ValueError("Recovery source snapshot differs")
    if (plan["source_snapshot"].get("run_id") != plan["source_run_id"]
            or info.get("source_run_id") != plan["source_run_id"]
            or info.get("preserved_candidates") != plan["preserved_candidates"]
            or manifest["completed_candidates"] < plan["preserved_candidates"]
            or info.get("extension_sha256") != plan["extension_sha256"]
            or manifest.get("protocol") != plan["protocol"]
            or manifest.get("task") != plan["task"]
            or type(plan["preserved_candidates"]) is not int
            or not 0 < plan["preserved_candidates"] < len(work)
            or plan["total_candidates"] != len(work)
            or work[plan["preserved_candidates"]][1:] != (plan["next_candidate_id"], plan["next_seed"])):
        raise ValueError("Recovery lineage/count/implementation differs")
    for name in ("candidates.jsonl", "costs.jsonl"):
        expected = plan["source_snapshot"]["files"][name]
        with (run / name).open("rb") as stream:
            prefix = stream.read(expected["bytes"])
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
        "task": info["manifest"]["task"], "verified_source_digest": snapshot_digest(info["snapshot"]), "preserved_candidates": count,
        "remaining_candidates": len(work) - count,
        "next_candidate_id": work[count][1] if count < len(work) else None,
        "next_seed": work[count][2] if count < len(work) else None,
        "correctness_scored": False, "gpu_or_api_work_started": False}


def build_plan(root, info):
    m = info["manifest"]
    if m.get("status") != "interrupted" or m.get("error_type") != "KeyboardInterrupt" or info["count"] >= len(info["work"]):
        raise ValueError("Manual recovery requires an interrupted, incomplete run; inspect failed/complete runs separately")
    return {"schema_version": 1, "kind": KIND, "task": m["task"],
        "source_run_id": info["snapshot"]["run_id"], "source_snapshot": info["snapshot"],
        "source_digest": snapshot_digest(info["snapshot"]),
        "protocol": info["protocol"], "extension_sha256": extension_hashes(root),
        "preserved_candidates": info["count"], "total_candidates": len(info["work"]),
        "next_candidate_id": info["work"][info["count"]][1],
        "next_seed": info["work"][info["count"]][2],
        "policy": "Operational recovery: preserve every saved record, generate missing candidates with original seeds, never filter by quality.",
        "limitation": "Candidate boundary recovery, not restoration of an interrupted diffusion trajectory; GPU bitwise equivalence is not asserted."}


def prepare_plan(root, run, destination=None):
    info = inspect_run(root, run)
    plan = build_plan(root, info)
    destination = destination or root / "artifacts/collection-recovery-plans" / (run.name + ".json")
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
    collection.profile_path(plan["task"])
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
    if plan != build_plan(root, info):
        raise ValueError("Recovery plan does not match the interrupted source")
    return plan, run, info


def recover(root, plan_path):
    plan, source, info = check_server_source(root, plan_path)
    config, work, preserved = info["config"], info["work"], info["count"]
    hardware = collection.runtime_check(root, config)
    dependencies = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    if dependencies != info["manifest"].get("dependencies"):
        raise ValueError("Runtime versions differ from the original source run")
    # One explicit attempt per immutable source: reject concurrent/accidental duplicate launches.
    claims = root / "artifacts/collection-recovery-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim = claims / (source.name + ".json")
    try:
        claim_stream = claim.open("x")
    except FileExistsError as exc:
        raise ValueError(f"Recovery was already attempted for this source. Inspect {claim}; after another interruption, archive the NEW run and prepare its own plan.") from exc
    with claim_stream:
        run = new_run(root / "artifacts/runs", f"fast-llada-{plan['task']}-recovery")
        json.dump({"run_id": run.name, "plan_sha256": frozen.sha256(plan_path)}, claim_stream)
        claim_stream.write("\n")
    print(f"Recovery run directory: {run}", flush=True)
    print(f"Preserving {preserved}/{len(work)} candidates; generating {len(work)-preserved} missing candidates", flush=True)
    m = {"status": "initializing", "mode": collection.SCOPE, "protocol": info["protocol"],
        "task": plan["task"], "candidates_per_question": 8, "evaluation_ready": False,
        "is_benchmark_result": False, "question_count": len(info["questions"]),
        "completed_candidates": preserved, "pid": os.getpid(), "backend": "fast_dllm_v1",
        **{key: info["manifest"][key] for key in ("incomplete_candidates", "truncated_candidates", "empty_candidates")},
        "physical_gpu_id": os.environ.get("GPU_ID"),
        "questions_sha256": config["datasets"]["questions_sha256"],
        "python": platform.python_version(), "platform": platform.platform(), "dependencies": dependencies,
        "gpu": hardware["gpu_name"], "hardware_at_start": hardware,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
        "warmup": "One handwritten unscored runtime example; not benchmark tuning or validation",
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
        for name in ("config.snapshot.json", "collection-profile.json", "generation-lock.json", "candidates.jsonl", "costs.jsonl"):
            shutil.copyfile(source / name, run / name)
        shutil.copyfile(source / "manifest.json", run / "source-manifest.json")
        shutil.copyfile(plan_path, run / "recovery-plan.json")
        check_rows(run, work, m)
        check_lineage(root, run, m, work)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (run / "dependencies.txt").write_text(freeze.stdout)
        m["dependency_capture_succeeded"] = True
        load_start = time.perf_counter()
        backend = collection.make_backend(root, config)
        m["model_load_seconds"] = time.perf_counter() - load_start
        warm_start = time.perf_counter()
        question = collection.warmup_question(plan["task"])
        candidate, cost = backend.generate(question, candidate_id="warmup", seed=config["runtime"]["master_seed"])
        collection.validate_record(candidate, cost, question, "warmup", config["runtime"]["master_seed"])
        m["warmup_flags"] = {"completed": candidate.completed, "truncated": candidate.truncated,
                             "empty": not candidate.text.strip()}
        m["warmup_seconds"] = time.perf_counter() - warm_start
        generation_start = time.perf_counter()
        m["status"] = "running"
        write_json(run / "manifest.json", m)
        with (run / "candidates.jsonl").open("a") as candidates, (run / "costs.jsonl").open("a") as costs:
            for question, candidate_id, seed in work[preserved:]:
                candidate, cost = backend.generate(question, candidate_id=candidate_id, seed=seed)
                collection.validate_record(candidate, cost, question, candidate_id, seed)
                cost_row = {"question_id": question.question_id, "candidate_id": candidate_id, **asdict(cost)}
                candidates.write(json.dumps(asdict(candidate), allow_nan=False) + "\n")
                costs.write(json.dumps(cost_row, allow_nan=False) + "\n")
                candidates.flush()
                costs.flush()
                m["completed_candidates"] += 1
                m["incomplete_candidates"] += int(not candidate.completed)
                m["truncated_candidates"] += int(candidate.truncated)
                m["empty_candidates"] += int(not candidate.text.strip())
                m["generation_elapsed_seconds"] = time.perf_counter() - generation_start
                write_json(run / "manifest.json", m)
                print(f"[{plan['task']}] Completed {m['completed_candidates']}/{len(work)} candidates (preserved {preserved})", flush=True)
        check_rows(run, work, m)
        check_lineage(root, run, m, work)
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
