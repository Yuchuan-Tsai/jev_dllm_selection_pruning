"""Pinned answer collection for MATH500/HumanEval, separate from validated scoring."""
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
from ..backends.fast_llada import FastLLaDA, verify_sources
from ..config import validate
from ..data.splits import _prompt_key
from ..operations import remote
from ..reproducibility import trajectory_seed
from ..schema import Candidate, CostRecord, Question

TASKS = {"math500": 500, "humaneval": 164}
SCOPE = "early_heldout_answer_collection_v1"
LOCK = "configs/collection/generation-lock-v5.json"
EXTENSION = (
    "Makefile", "scripts/collect_benchmarks.py",
    "src/jev_dllm/experiments/collection.py", "src/jev_dllm/operations/collection.py",
    "configs/collection/math500.json", "configs/collection/humaneval.json",
    "prompts/collection/math500.txt", "prompts/collection/humaneval.txt",
)
INPUTS = tuple(name for task in TASKS for name in
               (f"data/processed/{task}-heldout.json", f"data/manifests/{task}-week01.json"))


def profile_path(task):
    if task not in TASKS:
        raise ValueError("Choose math500 or humaneval")
    return f"configs/collection/{task}.json"


def locked_paths(root):
    original = json.loads((root / frozen.RECORD).read_text())
    return sorted(set(EXTENSION + INPUTS + (frozen.PROFILE, frozen.RECORD)
                      + tuple(original["public_files_sha256"])))


def check(root, task):
    path = profile_path(task)
    record = json.loads((root / LOCK).read_text())
    if record.get("scope") != SCOPE or set(record.get("files_sha256", {})) != set(locked_paths(root)):
        raise ValueError("Collection lock has an unexpected scope or file list")
    for name, expected in record["files_sha256"].items():
        p = root / name
        if p.is_symlink() or any((root / parent).is_symlink() for parent in Path(name).parents):
            raise ValueError("Pinned collection paths must not contain symlinks")
        if frozen.sha256(p) != expected:
            raise ValueError(f"Pinned collection file changed: {name}; do not refresh pins after generation")
    # Check the matching GSM8K implementation record before task-specific inputs.
    original = json.loads((root / frozen.PROFILE).read_text())
    frozen.check_protocol(root, original, root / original["datasets"]["questions_path"])
    spec = json.loads((root / path).read_text())
    if (spec.get("scope") != SCOPE or spec.get("task") != task
            or spec.get("question_count") != TASKS[task]
            or spec.get("evaluation_ready") is not False
            or spec.get("generation_settings_locked") is not True):
        raise ValueError("Collection profile must not claim validated evaluation")
    config = spec["backend_profile"]
    validate(config)
    g = config["generator"]
    if (config["frozen"] is not False or g["selected_backend"] != "fast_dllm_v1"
            or g["candidates_per_question"] != 8 or g["generation_length"] != 512
            or g["block_size"] != 32 or config["backends"]["fast_dllm_v1"]["steps"] != 512
            or config["runtime"]["hardware"] != "NVIDIA A100-SXM4-80GB"):
        raise ValueError("Collection requires the pinned N=8, length=512, block=32 A100 profile")
    d = config["datasets"]
    if (d["name"] != task or d["questions_path"] != f"data/processed/{task}-heldout.json"
            or d["manifest_path"] != f"data/manifests/{task}-week01.json"):
        raise ValueError("Wrong task input paths")
    if (frozen.sha256(root / d["questions_path"]) != d["questions_sha256"]
            or frozen.sha256(root / d["manifest_path"]) != d["manifest_sha256"]):
        raise ValueError("Task data differ from the profile")
    manifest = json.loads((root / d["manifest_path"]).read_text())
    questions = load_questions(root / d["questions_path"])
    if (len(questions) != TASKS[task] or [q.question_id for q in questions] != manifest["question_ids"]
            or any(q.dataset != task or q.split != "heldout" for q in questions)
            or len({_prompt_key(q.prompt) for q in questions}) != len(questions)):
        raise ValueError("Question count, order, uniqueness or task does not match the source manifest")
    prompt = config["prompts"]["generation_math"]  # Adapter's legacy key holds the task-specific text.
    if (prompt["path"] != f"prompts/collection/{task}.txt"
            or frozen.sha256(root / prompt["path"]) != prompt["sha256"]
            or (root / prompt["path"]).read_text().count("{question}") != 1):
        raise ValueError("Task-specific generation prompt differs")
    seeds = [trajectory_seed(config["runtime"]["master_seed"], q.question_id, i)
             for q in questions for i in range(8)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate planned trajectory seed")
    verify_sources(root, config)
    return spec, questions


def contract(root, task):
    spec, _ = check(root, task)
    return {"scope": SCOPE, "task": task, "generation_lock_sha256": frozen.sha256(root / LOCK),
            "profile_sha256": frozen.sha256(root / profile_path(task)),
            "evaluation_ready": spec["evaluation_ready"], "is_benchmark_result": False}


def validate_record(candidate, cost, question, cid, seed):
    if not isinstance(candidate, Candidate) or not isinstance(cost, CostRecord):
        raise ValueError("Invalid candidate/cost record types")
    if ((candidate.question_id, candidate.candidate_id, candidate.seed) != (question.question_id, cid, seed)
            or type(candidate.seed) is not int or not isinstance(candidate.text, str)
            or type(candidate.completed) is not bool or type(candidate.truncated) is not bool):
        raise ValueError("Unexpected candidate identity, seed or fields")
    if candidate.confidence is not None and (type(candidate.confidence) not in (int, float)
                                            or not math.isfinite(candidate.confidence)):
        raise ValueError("Invalid confidence")
    for name, value in asdict(cost).items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid cost: {name}")
    if type(cost.trajectory_nfe) is not int or not 1 <= cost.trajectory_nfe <= 512:
        raise ValueError("Invalid measured forward-call count")
    if any(getattr(cost, name) is None for name in
           ("generation_gpu_seconds", "generation_wall_seconds", "peak_memory_bytes")):
        raise ValueError("Missing generation measurement")


def runtime_check(root, config):
    from ..operations.dependencies import check_lock
    from ..operations.server import gpu_check
    import torch
    check_lock(root / "requirements/gpu-linux-py311.lock")
    hardware = gpu_check(root)
    frozen.check_hardware(config, torch)
    return hardware


def make_backend(root, config):
    return FastLLaDA(config, root, (root / config["prompts"]["generation_math"]["path"]).read_text(), False)


def warmup_question(task):
    # Handwritten runtime inputs, not benchmark examples or correctness validation.
    prompt = ("Compute 2 + 3." if task == "math500" else
              'def increment(value: int) -> int:\n    """Return value plus one."""\n')
    return Question(f"runtime/{task}/warmup", task, "fixture", prompt)


def generate(root, task):
    spec, questions = check(root, task)
    config = spec["backend_profile"]
    provenance = contract(root, task)
    hardware = runtime_check(root, config)
    print(json.dumps(hardware, indent=2), flush=True)
    run = new_run(root / "artifacts/runs", f"fast-llada-{task}")
    print(f"Run directory: {run}", flush=True)
    m = {"mode": SCOPE, "status": "initializing", "task": task, "protocol": provenance,
         "question_count": len(questions), "candidates_per_question": 8,
         "completed_candidates": 0, "incomplete_candidates": 0, "truncated_candidates": 0,
         "empty_candidates": 0, "is_benchmark_result": False, "evaluation_ready": False,
         "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
         "python": platform.python_version(), "platform": platform.platform(),
         "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
         "physical_gpu_id": os.environ.get("GPU_ID"), "hardware_at_start": hardware,
         "questions_sha256": config["datasets"]["questions_sha256"], "api_calls": 0,
         "warmup": "One handwritten unscored runtime example; not benchmark tuning or validation",
         "readiness_note": "User-requested early collection before separate development/scoring validation; keep raw candidates unchanged."}
    write_json(run / "manifest.json", m)
    started = time.perf_counter()
    generation_start = None
    try:
        shutil.copyfile(root / profile_path(task), run / "collection-profile.json")
        shutil.copyfile(root / LOCK, run / "generation-lock.json")
        write_json(run / "config.snapshot.json", config)
        deployment = root / "artifacts/collection-deployment.json"
        if deployment.exists():
            shutil.copyfile(deployment, run / "deployment.json")
            m["deployment_sha256"] = frozen.sha256(deployment)
            m["project_commit"] = json.loads(deployment.read_text()).get("project_commit")
        capture = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        (run / "dependencies.txt").write_text(capture.stdout)
        m["dependency_capture_succeeded"] = True
        load_start = time.perf_counter()
        backend = make_backend(root, config)
        m["model_load_seconds"] = time.perf_counter() - load_start
        warm_start = time.perf_counter()
        question = warmup_question(task)
        candidate, cost = backend.generate(question, candidate_id="warmup", seed=config["runtime"]["master_seed"])
        validate_record(candidate, cost, question, "warmup", config["runtime"]["master_seed"])
        m["warmup_seconds"] = time.perf_counter() - warm_start
        m["warmup_flags"] = {"completed": candidate.completed, "truncated": candidate.truncated,
                             "empty": not candidate.text.strip()}
        generation_start = time.perf_counter()
        m["status"] = "running"
        write_json(run / "manifest.json", m)
        with (run / "candidates.jsonl").open("x") as texts, (run / "costs.jsonl").open("x") as costs:
            for question in questions:
                for index in range(8):
                    cid = f"{question.question_id}/c{index:02}"
                    seed = trajectory_seed(config["runtime"]["master_seed"], question.question_id, index)
                    candidate, cost = backend.generate(question, candidate_id=cid, seed=seed)
                    validate_record(candidate, cost, question, cid, seed)
                    texts.write(json.dumps(asdict(candidate), allow_nan=False) + "\n")
                    costs.write(json.dumps({"question_id": question.question_id, "candidate_id": cid, **asdict(cost)}, allow_nan=False) + "\n")
                    texts.flush()
                    costs.flush()
                    m["completed_candidates"] += 1
                    m["incomplete_candidates"] += int(not candidate.completed)
                    m["truncated_candidates"] += int(candidate.truncated)
                    m["empty_candidates"] += int(not candidate.text.strip())
                    m["generation_elapsed_seconds"] = time.perf_counter() - generation_start
                    write_json(run / "manifest.json", m)
                    print(f"[{task}] Completed {m['completed_candidates']}/{len(questions)*8} candidates", flush=True)
        m["status"] = "complete"
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        m["total_seconds_including_load_and_warmup"] = time.perf_counter() - started
        if generation_start is not None:
            m["generation_phase_seconds"] = time.perf_counter() - generation_start
        write_json(run / "manifest.json", m)
    return run


def audit(root, run):
    """Inspect a transferred archive without private references or executing code."""
    remote.valid_run_id(run.name)
    receipt = json.loads((run.parent / ".receipts" / (run.name + ".json")).read_text())
    snapshot = receipt["snapshot"]
    if snapshot.get("run_id") != run.name or receipt.get("digest") != remote.snapshot_digest(snapshot):
        raise ValueError("Invalid archive receipt")
    remote.verify_copy(run.parent, snapshot)
    m = json.loads((run / "manifest.json").read_text())
    task = m["task"]
    spec, questions = check(root, task)
    if (m.get("mode") != SCOPE or m.get("protocol") != contract(root, task)
            or m.get("status") not in remote.TERMINAL
            or m.get("question_count") != TASKS[task]
            or m.get("questions_sha256") != spec["backend_profile"]["datasets"]["questions_sha256"]
            or json.loads((run / "collection-profile.json").read_text()) != spec
            or json.loads((run / "config.snapshot.json").read_text()) != spec["backend_profile"]
            or frozen.sha256(run / "generation-lock.json") != frozen.sha256(root / LOCK)):
        raise ValueError("Archive is not from this pinned collection")
    def rows(name):
        raw = (run / name).read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("Partial final record; inspect before analysis")
        return [json.loads(line) for line in raw.splitlines()]
    candidates, costs = rows("candidates.jsonl"), rows("costs.jsonl")
    count = m["completed_candidates"]
    total = TASKS[task] * 8
    if (type(count) is not int or len(candidates) != count or len(costs) != count
            or not 0 <= count <= total or (m["status"] == "complete" and count != total)):
        raise ValueError("Archive counts disagree")
    flags = {"incomplete_candidates": 0, "truncated_candidates": 0, "empty_candidates": 0}
    for index, (row, cost) in enumerate(zip(candidates, costs)):
        q = questions[index // 8]
        cid = f"{q.question_id}/c{index%8:02}"
        seed = trajectory_seed(spec["backend_profile"]["runtime"]["master_seed"], q.question_id, index % 8)
        if (set(cost) != {f.name for f in fields(CostRecord)} | {"question_id", "candidate_id"}
                or (cost["question_id"], cost["candidate_id"]) != (q.question_id, cid)):
            raise ValueError("Mismatched cost record")
        candidate = Candidate(**row)
        validate_record(candidate, CostRecord(**{f.name: cost[f.name] for f in fields(CostRecord)}), q, cid, seed)
        flags["incomplete_candidates"] += int(not candidate.completed)
        flags["truncated_candidates"] += int(candidate.truncated)
        flags["empty_candidates"] += int(not candidate.text.strip())
    if any(m.get(key) != value for key, value in flags.items()):
        raise ValueError("Manifest completion flags disagree with saved records")
    return {"run_id": run.name, "task": task, "status": m["status"], "saved_candidates": count,
            "expected_candidates": total, **flags, "correctness_scored": False,
            "jev_requests_sent": 0, "generation_settings_verified": True, "evaluation_ready": False}
