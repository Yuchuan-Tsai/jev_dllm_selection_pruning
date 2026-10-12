"""Generate development pools or the explicitly frozen GSM8K-500 pool."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from ..backends.official_llada import OfficialLLaDA
from ..backends.fast_llada import FastLLaDA
from ..reproducibility import trajectory_seed
from .selection import load_questions, new_run, write_json
from . import frozen


def generate_pool(root: Path, config: dict, questions_path: Path, output: Path,
                  allow_download: bool = False) -> Path:
    questions = load_questions(questions_path)
    protocol = frozen.check_protocol(root, config, questions_path) if frozen.required(config, questions) else None
    question_pin = config.get("datasets", {}).get("questions_sha256")
    if question_pin and hashlib.sha256(questions_path.read_bytes()).hexdigest() != question_pin:
        raise ValueError("Questions differ from the pinned generation input")
    if any(q.dataset != "gsm8k" for q in questions):
        raise ValueError("The active generation profile is GSM8K-only; other benchmarks require task-specific validated profiles")
    if not protocol and any(q.split != "dev" for q in questions):
        raise ValueError("This initial generator runner is development-only; held-out runs require a frozen protocol")
    prompt_path = root / config["prompts"]["generation_math"]["path"]
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != config["prompts"]["generation_math"]["sha256"]:
        raise ValueError("Generation prompt differs from the recorded configuration")
    selected_backend = config["generator"]["selected_backend"]
    if protocol:
        from ..operations.dependencies import check_lock
        check_lock(root / "requirements/gpu-linux-py311.lock")
        import torch
        frozen.check_hardware(config, torch)
    prefix = "fast-llada-gsm8k500" if protocol else ("fast-llada-dev" if selected_backend == "fast_dllm_v1" else "llada-dev")
    run_dir = new_run(output, prefix)
    print(f"Run directory: {run_dir}", flush=True)
    write_json(run_dir / "config.snapshot.json", config)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    diff = subprocess.run(["git", "diff", "HEAD", "--"], cwd=root, capture_output=True, check=False)
    manifest = {"status": "initializing", "mode": frozen.PHASE if protocol else "development_pilot", "is_benchmark_result": False,
                "protocol": protocol,
                "question_count": len(questions), "completed_candidates": 0, "pid": os.getpid(),
                "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
                "python": platform.python_version(), "platform": platform.platform(),
                "project_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
                "warmup": "one unscored call on the first input question", "api_calls": 0,
                "backend": selected_backend}
    deployment = root / "artifacts/deployment.json"
    if deployment.exists():
        deployed = json.loads(deployment.read_text())
        write_json(run_dir / "deployment.json", deployed)
        manifest["deployment_sha256"] = hashlib.sha256(deployment.read_bytes()).hexdigest()
        if manifest["project_commit"] is None:
            manifest["project_commit"] = deployed.get("project_commit")
    write_json(run_dir / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        if selected_backend == "official_llada":
            backend = OfficialLLaDA(config, root / "third_party/llada/generate.py", prompt_path.read_text(), allow_download)
        elif selected_backend == "fast_dllm_v1":
            backend = FastLLaDA(config, root, prompt_path.read_text(), allow_download)
        else:
            raise ValueError("Unsupported generation backend")
        manifest["model_load_seconds"] = time.perf_counter() - started
        manifest["gpu"] = backend.torch.cuda.get_device_name(0)
        manifest["dependencies"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        (run_dir / "dependencies.txt").write_text(freeze.stdout)
        manifest["dependency_capture_succeeded"] = freeze.returncode == 0
        backend.generate(questions[0], candidate_id="warmup", seed=config["runtime"]["master_seed"])
        generation_start = time.perf_counter()
        manifest["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)
        with (run_dir / "candidates.jsonl").open("w") as candidates, (run_dir / "costs.jsonl").open("w") as costs:
            for question in questions:
                for index in range(config["generator"]["candidates_per_question"]):
                    seed = trajectory_seed(config["runtime"]["master_seed"], question.question_id, index)
                    candidate, cost = backend.generate(question, candidate_id=f"{question.question_id}/c{index:02}", seed=seed)
                    candidates.write(json.dumps(asdict(candidate)) + "\n")
                    costs.write(json.dumps({"question_id": question.question_id, "candidate_id": candidate.candidate_id, **asdict(cost)}) + "\n")
                    candidates.flush(); costs.flush()
                    manifest["completed_candidates"] += 1
                    write_json(run_dir / "manifest.json", manifest)
                print(f"Completed {manifest['completed_candidates']}/{len(questions) * config['generator']['candidates_per_question']} candidates", flush=True)
        manifest.update({"status": "complete", "generation_phase_seconds": time.perf_counter() - generation_start})
    except BaseException as exc:
        manifest.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error_type": type(exc).__name__})
        raise
    finally:
        manifest["total_seconds_including_load_and_warmup"] = time.perf_counter() - started
        write_json(run_dir / "manifest.json", manifest)
    return run_dir
