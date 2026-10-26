"""Week 2 development pools: N instrumented Fast-dLLM candidates per question, resumable."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .collection import validate_record
from .selection import load_questions, new_run, write_json
from ..backends.instrumented import checkpoint_blocks
from ..operations import remote
from ..reproducibility import fingerprint, trajectory_seed
from ..schema import Candidate, CostRecord

SCOPE = "week02_development_pools"
TASKS = ("gsm8k", "math", "mbpp")
FILES = ("candidates.jsonl", "costs.jsonl", "trajectories.jsonl")
# Equivalence reruns the Week 1 Fast-dLLM pilot exactly, recording trajectories.
EQUIVALENCE = {"base_profile": "configs/week01/fast-diversity.json", "questions": "data/processed/gsm8k-dev.json",
               "checkpoints": [2, 4, 6]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_path(task):
    if task not in TASKS:
        raise ValueError("Choose gsm8k, math or mbpp")
    return f"configs/week02/pool-{task}.json"


def load_profile(root: Path, task: str):
    profile = json.loads((root / profile_path(task)).read_text())
    if profile.get("scope") != SCOPE or profile.get("task") != task:
        raise ValueError("Unexpected Week 2 pool profile")
    n = profile["candidates_per_question"]
    if type(n) is not int or not 2 <= n <= 255:
        raise ValueError("candidates_per_question must be an integer from 2 to 255")
    base_spec, question_spec = profile["base_profile"], profile["questions"]
    if sha256(root / base_spec["path"]) != base_spec["sha256"]:
        raise ValueError("Base generation profile differs from its pin")
    base = json.loads((root / base_spec["path"]).read_text())
    base = base[base_spec["key"]] if base_spec["key"] else base
    if sha256(root / question_spec["path"]) != question_spec["sha256"]:
        raise ValueError("Development questions differ from the profile pin; run the Week 2 data upload")
    questions = load_questions(root / question_spec["path"])
    if len(questions) != question_spec["count"] or any(q.dataset != task or q.split != "dev" for q in questions):
        raise ValueError("Pool inputs must be the pinned development questions for this task")
    check_settings(root, base, profile["checkpoint_completed_blocks"])
    return profile, base, questions


def check_settings(root, base, checkpoints):
    g = base["generator"]
    if g["selected_backend"] != "fast_dllm_v1" or g["generation_length"] % g["block_size"]:
        raise ValueError("Week 2 pools use the Week 1 Fast-dLLM profile")
    checkpoint_blocks(g["generation_length"] // g["block_size"], checkpoints)
    prompt = base["prompts"]["generation_math"]  # Legacy key holds the task-specific prompt.
    if sha256(root / prompt["path"]) != prompt["sha256"]:
        raise ValueError("Generation prompt differs from the base profile")


def runtime(root):
    from ..operations.dependencies import check_lock
    from ..operations.server import gpu_check
    check_lock(root / "requirements/gpu-linux-py311.lock")
    return gpu_check(root)


def make_backend(root, base, checkpoints):
    from ..backends.instrumented import InstrumentedFastLLaDA
    prompt = (root / base["prompts"]["generation_math"]["path"]).read_text()
    return InstrumentedFastLLaDA(base, root, prompt, checkpoints, False)


def plan(questions, n, master_seed):
    return [(q, f"{q.question_id}/c{i:02}", trajectory_seed(master_seed, q.question_id, i))
            for q in questions for i in range(n)]


def read_rows(path: Path) -> list:
    raw = path.read_bytes()
    # A trailing partial line from an interrupted write is not a saved record.
    lines = raw.split(b"\n")
    return [json.loads(line) for line in (lines if raw.endswith(b"\n") else lines[:-1]) if line]


def saved_prefix(source: Path, schedule, profile_bytes: bytes):
    remote.valid_run_id(source.name)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("mode") != SCOPE or manifest.get("status") not in {"failed", "interrupted"}:
        raise ValueError("Resume only from a failed or interrupted Week 2 pool run")
    if (source / "pool-profile.json").read_bytes() != profile_bytes:
        raise ValueError("Source run used a different pool profile")
    rows = [read_rows(source / name) for name in FILES]
    count = min(len(r) for r in rows)
    for (question, cid, seed), candidate, cost, trace in zip(schedule, *(r[:count] for r in rows)):
        if ((candidate["candidate_id"], candidate["seed"], cost["candidate_id"], trace["candidate_id"], trace["seed"])
                != (cid, seed, cid, cid, seed) or candidate["question_id"] != question.question_id):
            raise ValueError("Saved prefix does not follow the planned order and seeds")
    return [r[:count] for r in rows], {"run_id": source.name, "copied_candidates": count,
                                       "source_sha256": {name: sha256(source / name) for name in FILES}}


def write_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()


def generate_into(run, manifest, backend, schedule, warmup, prefix=None):
    """Generate the remaining schedule after an optional verified prefix; flush every record."""
    started, generation_start = time.perf_counter(), None
    try:
        streams = [(run / name).open("x") for name in FILES]
        try:
            for rows in zip(*(prefix or ([], [], []))):
                for stream, row in zip(streams, rows):
                    write_line(stream, row)
                manifest["completed_candidates"] += 1
                manifest["truncated_candidates"] += int(rows[0]["truncated"])
                manifest["empty_candidates"] += int(not rows[0]["text"].strip())
            question, seed = warmup
            warm_start = time.perf_counter()
            backend.generate(question, candidate_id="warmup", seed=seed)
            manifest["warmup_seconds"] = time.perf_counter() - warm_start
            manifest["status"] = "running"
            write_json(run / "manifest.json", manifest)
            generation_start = time.perf_counter()
            for question, cid, seed in schedule[manifest["completed_candidates"]:]:
                candidate, cost = backend.generate(question, candidate_id=cid, seed=seed)
                validate_record(candidate, cost, question, cid, seed)
                trace = backend.last_trace
                if (trace["candidate_id"], trace["seed"]) != (cid, seed):
                    raise ValueError("Trajectory record does not belong to this candidate")
                write_line(streams[0], asdict(candidate))
                write_line(streams[1], {"question_id": question.question_id, "candidate_id": cid, **asdict(cost)})
                write_line(streams[2], trace)
                manifest["completed_candidates"] += 1
                manifest["truncated_candidates"] += int(candidate.truncated)
                manifest["empty_candidates"] += int(not candidate.text.strip())
                manifest["generation_elapsed_seconds"] = time.perf_counter() - generation_start
                write_json(run / "manifest.json", manifest)
                print(f"Completed {manifest['completed_candidates']}/{len(schedule)} candidates", flush=True)
        finally:
            for stream in streams:
                stream.close()
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__)
        raise
    finally:
        manifest["total_seconds_including_load_and_warmup"] = time.perf_counter() - started
        write_json(run / "manifest.json", manifest)


def start_run(root, prefix, record, hardware, backend_factory):
    run = new_run(root / "artifacts/runs", prefix)
    print(f"Run directory: {run}", flush=True)
    manifest = {"mode": SCOPE, "status": "initializing", **record, "completed_candidates": 0,
                "truncated_candidates": 0, "empty_candidates": 0, "is_benchmark_result": False,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                "python": platform.python_version(), "platform": platform.platform(),
                "hardware_at_start": hardware, "physical_gpu_id": os.environ.get("GPU_ID"), "api_calls": 0,
                # Free-text run conditions, e.g. tasks sharing one GPU, which inflate wall and GPU seconds.
                "run_note": os.environ.get("JEV_RUN_NOTE")}
    for name in ("artifacts/deployment.json", "artifacts/week2-deployment.json"):
        if (root / name).exists():
            manifest[Path(name).stem + "_sha256"] = sha256(root / name)
            manifest.setdefault("project_commit", json.loads((root / name).read_text()).get("project_commit"))
    write_json(run / "manifest.json", manifest)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    (run / "dependencies.txt").write_text(freeze.stdout)
    try:
        manifest["dependencies"] = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")}
    except importlib.metadata.PackageNotFoundError:
        manifest["dependencies"] = None
    load_start = time.perf_counter()
    backend = backend_factory()
    manifest["model_load_seconds"] = time.perf_counter() - load_start
    return run, manifest, backend


def generate(root: Path, task: str, resume_from: Path | None = None, backend_factory=None, runtime_check=runtime):
    profile, base, questions = load_profile(root, task)
    profile_bytes = (root / profile_path(task)).read_bytes()
    master_seed = base["runtime"]["master_seed"]
    schedule = plan(questions, profile["candidates_per_question"], master_seed)
    prefix, resumed = saved_prefix(resume_from, schedule, profile_bytes) if resume_from else (None, None)
    hardware = runtime_check(root)
    factory = backend_factory or (lambda: make_backend(root, base, profile["checkpoint_completed_blocks"]))
    record = {"task": task, "question_count": len(questions), "candidates_per_question": profile["candidates_per_question"],
              "pool_profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
              "base_profile_sha256": profile["base_profile"]["sha256"], "questions_sha256": profile["questions"]["sha256"],
              "base_profile_fingerprint": fingerprint(base), "master_seed": master_seed, "resumed_from": resumed,
              "warmup": "One unscored call on the first question with the master seed, as in Week 1 generation"}
    run, manifest, backend = start_run(root, f"w2pool-{task}", record, hardware, factory)
    (run / "pool-profile.json").write_bytes(profile_bytes)
    write_json(run / "base-profile.json", base)
    generate_into(run, manifest, backend, schedule, (questions[0], master_seed), prefix)
    return run


def equivalence(root: Path, reference: Path, backend_factory=None, runtime_check=runtime):
    """Rerun a Week 1 Fast-dLLM pilot with recording and compare every candidate to it."""
    base_path, questions_path = root / EQUIVALENCE["base_profile"], root / EQUIVALENCE["questions"]
    base = json.loads(base_path.read_text())
    check_settings(root, base, EQUIVALENCE["checkpoints"])
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    if (reference_manifest.get("status") != "complete" or reference_manifest.get("backend") != "fast_dllm_v1"
            or reference_manifest.get("questions_sha256") != sha256(questions_path)
            or fingerprint(json.loads((reference / "config.snapshot.json").read_text())) != fingerprint(base)):
        raise ValueError("Reference must be a complete Week 1 fast-diversity run on the pilot questions")
    questions = load_questions(questions_path)
    master_seed = base["runtime"]["master_seed"]
    schedule = plan(questions, base["generator"]["candidates_per_question"], master_seed)
    expected = read_rows(reference / "candidates.jsonl")
    if [(r["candidate_id"], r["seed"]) for r in expected] != [(cid, seed) for _, cid, seed in schedule]:
        raise ValueError("Reference candidates do not follow the planned order and seeds")
    hardware = runtime_check(root)
    factory = backend_factory or (lambda: make_backend(root, base, EQUIVALENCE["checkpoints"]))
    record = {"task": "equivalence", "question_count": len(questions),
              "candidates_per_question": base["generator"]["candidates_per_question"],
              "base_profile_sha256": sha256(base_path), "questions_sha256": sha256(questions_path),
              "reference_run": reference.name, "reference_candidates_sha256": sha256(reference / "candidates.jsonl"),
              "master_seed": master_seed}
    run, manifest, backend = start_run(root, "w2-equivalence", record, hardware, factory)
    generate_into(run, manifest, backend, schedule, (questions[0], master_seed))
    produced = read_rows(run / "candidates.jsonl")
    fields = ("text", "completed", "truncated")
    mismatches = [{"candidate_id": a["candidate_id"], "fields": [f for f in fields if a[f] != b[f]]}
                  for a, b in zip(produced, expected) if any(a[f] != b[f] for f in fields)]
    report = {"status": "identical" if not mismatches else "different", "compared": len(produced),
              "identical": len(produced) - len(mismatches), "mismatches": mismatches,
              "reference_gpu": reference_manifest.get("gpu"), "gpu": hardware.get("gpu_name"),
              "note": "Identical text under identical seeds shows that recording does not alter sampling on this hardware."}
    write_json(run / "equivalence.json", report)
    return run, report
