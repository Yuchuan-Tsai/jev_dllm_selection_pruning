"""Week 4 Qwen3-14B policy pool setup, generation and development comparison."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.backends.qwen3_policy import Qwen3Policy, profile
from jev_dllm.evaluation import week2_labels
from jev_dllm.experiments.selection import load_questions, new_run, write_json
from jev_dllm.experiments import week2_jev
from jev_dllm.operations import server
from jev_dllm.operations import remote
from jev_dllm.reproducibility import trajectory_seed
from jev_dllm.schema import Candidate
from jev_dllm.selection import week3 as select
from jev_dllm.selection.subsets import SIZES, nested_subsets
from jev_dllm.evaluation.week3_metrics import compare


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_model(root):
    frozen = profile(root)
    os.environ.update(server.gpu_environment(root))
    from huggingface_hub import snapshot_download
    policy = frozen["policy"]
    local = snapshot_download(policy["model_id"], revision=policy["revision"],
                              allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken"])
    return {"status": "cached", "model": policy["model_id"], "revision": policy["revision"], "snapshot": local}


def read_rows(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete final row: {path}")
    return [json.loads(line) for line in raw.splitlines() if line]


def verify_pool_copy(pool_dir):
    receipt_path = pool_dir.parent / ".receipts" / f"{pool_dir.name}.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["snapshot"].get("run_id") != pool_dir.name
            or receipt["digest"] != remote.snapshot_digest(receipt["snapshot"])):
        raise ValueError("Pool copy has no valid results-verify receipt")
    remote.verify_copy(pool_dir.parent, receipt["snapshot"])


def generate(root, task, limit=None, resume_from=None):
    frozen = profile(root)
    spec = frozen["policy"]["tasks"][task]
    questions = load_questions(root / spec["questions"])
    if limit is not None:
        if not 1 <= limit <= len(questions):
            raise ValueError("--limit must be between 1 and the development-set size")
        questions = questions[:limit]
    schedule = [(q, f"{q.question_id}/c{i:02}",
                 trajectory_seed(frozen["policy"]["master_seed"], q.question_id, i))
                for q in questions for i in range(frozen["policy"]["candidates_per_question"])]
    prefix = ([], [])
    resumed_from = None
    if resume_from is not None:
        source = (root / resume_from).resolve()
        if not source.is_relative_to((root / "artifacts/runs").resolve()):
            raise ValueError("Resume source must be a run under artifacts/runs")
        source_manifest = json.loads((source / "manifest.json").read_text())
        if (source_manifest.get("mode") != "week04_stronger_policy_development"
                or source_manifest.get("status") not in {"failed", "interrupted"}
                or source_manifest.get("task") != task
                or source_manifest.get("profile_sha256") != sha256(root / "configs/week04/qwen3-14b-policy.json")
                or source_manifest.get("question_count") != len(questions)
                or (source / "profile.json").read_bytes() != (root / "configs/week04/qwen3-14b-policy.json").read_bytes()):
            raise ValueError("Resume source does not match this task, question count and frozen profile")
        candidate_prefix, cost_prefix = read_rows(source / "candidates.jsonl"), read_rows(source / "costs.jsonl")
        if len(candidate_prefix) != len(cost_prefix) or len(candidate_prefix) >= len(schedule):
            raise ValueError("Resume source must have a matching incomplete candidate and cost prefix")
        for (question, cid, seed), candidate, cost in zip(schedule, candidate_prefix, cost_prefix):
            if ((candidate.get("question_id"), candidate.get("candidate_id"), candidate.get("seed")) !=
                    (question.question_id, cid, seed) or cost.get("candidate_id") != cid):
                raise ValueError("Resume files do not match the frozen candidate schedule")
        prefix = candidate_prefix, cost_prefix
        resumed_from = {"run_id": source.name, "candidates": len(candidate_prefix),
                        "candidates_sha256": sha256(source / "candidates.jsonl"),
                        "costs_sha256": sha256(source / "costs.jsonl")}
    hardware = server.gpu_check(root)
    policy = frozen["policy"]
    run = new_run(root / "artifacts/runs", f"w4-qwen3-14b-{task}")
    prompt_template = (root / spec["prompt"]).read_text()
    model_start = time.perf_counter()
    backend = Qwen3Policy(policy)
    manifest = {
        "mode": "week04_stronger_policy_development", "status": "running", "task": task,
        "model_id": policy["model_id"], "model_revision": policy["revision"],
        "thinking": policy["thinking"], "sampling": policy["sampling"],
        "max_new_tokens": spec["max_new_tokens"], "candidates_per_question": policy["candidates_per_question"],
        "question_count": len(questions), "questions_sha256": sha256(root / spec["questions"]),
        "profile_sha256": sha256(root / "configs/week04/qwen3-14b-policy.json"),
        "experiment_code_sha256": {name: sha256(root / name) for name in
                                    ("scripts/week4_qwen.py", "src/jev_dllm/backends/qwen3_policy.py",
                                     "src/jev_dllm/reproducibility.py")},
        "prompt_sha256": sha256(root / spec["prompt"]), "hardware": hardware,
        "model_load_seconds": time.perf_counter() - model_start, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "project_commit": os.environ.get("JEV_PROJECT_COMMIT"),
        "completed_candidates": 0, "api_calls": 0, "is_benchmark_result": False,
        "resumed_from": resumed_from,
        "note": "Development-only Qwen autoregressive generation; NFE and diffusion checkpoints do not apply."
    }
    write_json(run / "manifest.json", manifest)
    (run / "profile.json").write_bytes((root / "configs/week04/qwen3-14b-policy.json").read_bytes())
    with (run / "candidates.jsonl").open("x") as candidates, (run / "costs.jsonl").open("x") as costs:
        try:
            for c_row, cost_row in zip(*prefix):
                candidates.write(json.dumps(c_row, ensure_ascii=False, allow_nan=False) + "\n")
                costs.write(json.dumps(cost_row, allow_nan=False) + "\n")
            candidates.flush()
            costs.flush()
            manifest["completed_candidates"] = len(prefix[0])
            started = time.perf_counter()
            for question, cid, seed in schedule[manifest["completed_candidates"]:]:
                candidate, cost = backend.generate(question, prompt_template, candidate_id=cid, seed=seed,
                                                   max_new_tokens=spec["max_new_tokens"])
                candidates.write(json.dumps(asdict(candidate), ensure_ascii=False, allow_nan=False) + "\n")
                costs.write(json.dumps({"question_id": question.question_id, "candidate_id": cid,
                                        **asdict(cost)}, allow_nan=False) + "\n")
                candidates.flush()
                costs.flush()
                manifest["completed_candidates"] += 1
                manifest["generation_elapsed_seconds"] = time.perf_counter() - started
                write_json(run / "manifest.json", manifest)
                print(f"Completed {manifest['completed_candidates']}/{len(schedule)}", flush=True)
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error_type=type(exc).__name__)
            raise
        finally:
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            if manifest["status"] == "complete":
                manifest["files_sha256"] = {name: sha256(run / name) for name in
                                             ("candidates.jsonl", "costs.jsonl", "profile.json")}
            write_json(run / "manifest.json", manifest)
    return run


def prepare_jev(root, task, pool_dir, first):
    frozen = profile(root)
    spec = frozen["policy"]["tasks"][task]
    manifest = json.loads((pool_dir / "manifest.json").read_text())
    if manifest.get("mode") != "week04_stronger_policy_development" or manifest.get("status") != "complete" or manifest.get("task") != task:
        raise ValueError("Jev preparation requires a complete Week 4 Qwen pool for this task")
    if manifest.get("question_count") != spec["question_count"] or manifest.get("completed_candidates") != spec["question_count"] * frozen["policy"]["candidates_per_question"]:
        raise ValueError("Jev preparation requires the complete 200-question Qwen pool, not a pilot")
    verify_pool_copy(pool_dir)
    questions = load_questions(root / spec["questions"])
    if manifest.get("questions_sha256") != sha256(root / spec["questions"]):
        raise ValueError("Pool question input differs from the pinned development set")
    if not 1 <= first <= len(questions):
        raise ValueError("--first must select at least one development question")
    return week2_jev.prepare(root, task, pool_dir, root / spec["questions"],
                             [q.question_id for q in questions[:first]])


def analyze(root, task, pool_dir, jev_plan_dir, math_python=None):
    frozen = profile(root)
    spec = frozen["policy"]["tasks"][task]
    manifest = json.loads((pool_dir / "manifest.json").read_text())
    if (manifest.get("mode") != "week04_stronger_policy_development" or manifest.get("status") != "complete"
            or manifest.get("task") != task or manifest.get("profile_sha256") != sha256(root / "configs/week04/qwen3-14b-policy.json")):
        raise ValueError("Analysis requires a complete pool from the pinned Qwen policy profile")
    verify_pool_copy(pool_dir)
    if manifest.get("experiment_code_sha256") != {name: sha256(root / name) for name in
            ("scripts/week4_qwen.py", "src/jev_dllm/backends/qwen3_policy.py", "src/jev_dllm/reproducibility.py") }:
        raise ValueError("Week 4 Qwen generation implementation differs from the archived code pin")
    if manifest.get("files_sha256") != {name: sha256(pool_dir / name) for name in
            ("candidates.jsonl", "costs.jsonl", "profile.json")}:
        raise ValueError("Archived Qwen pool files changed after generation")
    questions = load_questions(root / spec["questions"])
    candidate_rows = read_rows(pool_dir / "candidates.jsonl")
    pools = {q.question_id: [] for q in questions}
    expected = [(q, f"{q.question_id}/c{i:02}", trajectory_seed(frozen["policy"]["master_seed"], q.question_id, i))
                for q in questions for i in range(frozen["policy"]["candidates_per_question"])]
    if len(candidate_rows) != len(expected):
        raise ValueError("Candidate file is incomplete")
    for (q, cid, seed), row in zip(expected, candidate_rows):
        c = Candidate(**row)
        if (c.question_id, c.candidate_id, c.seed) != (q.question_id, cid, seed):
            raise ValueError("Candidate file order or seeds differ from the frozen schedule")
        pools[q.question_id].append(c)
    refs = week2_labels.references(root, task, questions)
    out = new_run(root / "artifacts/runs", f"w4-qwen3-analysis-{task}")
    (out / "evaluation_private").mkdir()
    if task == "gsm8k":
        label_rows, details = week2_labels.gsm8k_labels(questions, pools, refs)
    elif task == "math":
        if math_python is None:
            raise ValueError("MATH analysis needs the isolated Math-Verify Python interpreter")
        label_rows, details = week2_labels.math_labels(root, math_python, questions, pools, refs, out)
    else:
        label_rows, details = week2_labels.mbpp_labels(root, questions, pools, refs, out)
    correct = {r["candidate_id"]: r["correct"] for r in label_rows}
    jev, _ = week2_jev.check(root, jev_plan_dir)
    jev_output = next((jev_plan_dir / name for name in ("live", "live-continued")
                       if (jev_plan_dir / name / "manifest.json").exists()
                       and json.loads((jev_plan_dir / name / "manifest.json").read_text()).get("status") == "complete"), None)
    if jev_output is None or jev.get("task") != task or jev.get("candidates_sha256") != sha256(pool_dir / "candidates.jsonl"):
        raise ValueError("Jev plan must be complete and cover this exact Qwen candidate pool")
    pointwise = {r["candidate_id"]: r["correct_probability"] for r in read_rows(jev_output / "results.jsonl")
                 if r.get("kind") == "pointwise"}
    if set(pointwise) != set(correct):
        raise ValueError("Jev pointwise scores must cover every candidate")
    by_n = {}
    available_sizes = tuple(n for n in SIZES if n <= len(pools[questions[0].question_id]))
    for n in available_sizes:
        subsets = {q.question_id: nested_subsets([c.candidate_id for c in pools[q.question_id]], (n,))[n]
                   for q in questions}
        picks = {"first": {qid: select.first(ids) for qid, ids in subsets.items()},
                 "seeded_random": {qid: select.seeded_random(qid, ids, week2_jev.SEED) for qid, ids in subsets.items()},
                 "jev_pointwise": {qid: select.highest(ids, pointwise) for qid, ids in subsets.items()}}
        by_n[str(n)] = compare(picks, subsets, correct, week2_jev.SEED)
    metrics = {"task": task, "policy": manifest["model_id"], "questions": len(questions),
               "by_subset_size": by_n, "jev_plan": jev_plan_dir.name, "jev_output": jev_output.name,
               "pool_run": pool_dir.name, "scorer_details": details,
               "generation_cost": {"mean_gpu_seconds": sum(r["generation_gpu_seconds"] or 0 for r in read_rows(pool_dir / "costs.jsonl")) / len(candidate_rows),
                                   "max_new_tokens": spec["max_new_tokens"]},
               "note": "Week 4 development comparison only; Jev and random/first are evaluated on the same Qwen3-14B candidate pool."}
    write_json(out / "metrics.json", metrics)
    (out / "summary.md").write_text("\n".join([f"Task: {task}; policy: {manifest['model_id']}", "",
        "| Selector | " + " | ".join(f"N={n}" for n in by_n) + " |",
        "|---|" + "---|" * len(by_n)] + [
        f"| {name} | " + " | ".join(f"{by_n[n]['selectors'][name]['accuracy']:.1%}" for n in by_n) + " |"
        for name in ("first", "seeded_random", "jev_pointwise")]) + "\n")
    write_json(out / "manifest.json", {"status": "complete", "task": task, "pool_run": pool_dir.name,
                                         "metrics_sha256": sha256(out / "metrics.json"),
                                         "finished_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0})
    (out / "evaluation_private/labels.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in label_rows))
    return out


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("check")
    commands.add_parser("fetch")
    cmd = commands.add_parser("generate")
    cmd.add_argument("--task", choices=("gsm8k", "math", "mbpp"), required=True)
    cmd.add_argument("--limit", type=int)
    cmd.add_argument("--resume-from", type=Path)
    cmd = commands.add_parser("jev-prepare")
    cmd.add_argument("--task", choices=("gsm8k", "math", "mbpp"), required=True)
    cmd.add_argument("--pool-dir", type=Path, required=True)
    cmd.add_argument("--first", type=int, default=200)
    cmd = commands.add_parser("analyze")
    cmd.add_argument("--task", choices=("gsm8k", "math", "mbpp"), required=True)
    cmd.add_argument("--pool-dir", type=Path, required=True)
    cmd.add_argument("--jev-plan", type=Path, required=True)
    cmd.add_argument("--math-python", type=Path, default=Path(".venv-math/bin/python"))
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "check":
            record = profile(root)
            import torch
            if torch.__version__.split("+")[0] != "2.5.1" or not (torch.version.cuda or "").startswith("12."):
                raise RuntimeError(f"Expected the A800 PyTorch 2.5.1 / CUDA 12 image; found torch={torch.__version__}, cuda={torch.version.cuda}")
            result = {"status": "passed", "model": record["policy"]["model_id"],
                      "revision": record["policy"]["revision"], "tasks": list(record["policy"]["tasks"]),
                      "torch": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_work_started": False}
        elif args.action == "fetch":
            result = fetch_model(root)
        elif args.action == "generate":
            os.environ.update(server.gpu_environment(root))
            result = {"status": "complete", "run_directory": str(generate(root, args.task, args.limit, args.resume_from))}
        elif args.action == "jev-prepare":
            out = prepare_jev(root, args.task, (root / args.pool_dir).resolve(), args.first)
            result = {"status": "prepared", "plan_directory": str(out),
                      "planned_calls": json.loads((out / "plan.json").read_text())["planned_calls"], "paid_calls_sent": 0}
        else:
            math_python = (root / args.math_python) if args.task == "math" else None
            out = analyze(root, args.task, (root / args.pool_dir).resolve(), (root / args.jev_plan).resolve(), math_python)
            print((out / "summary.md").read_text())
            result = {"status": "complete", "analysis_directory": str(out)}
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print(f"Week 4 Qwen command stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
