"""Freeze and evaluate the completed frozen-pointwise Jev MATH500 run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_dllm.evaluation import math500 as scorer
from jev_dllm.experiments import math500_jev, selection
from jev_dllm.reproducibility import trajectory_seed
from jev_dllm.selection.baselines import random_choice


def digest(path):
    return scorer.sha256(path)


def inventory(path):
    path = Path(path)
    return {p.relative_to(path).as_posix(): digest(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def load(path):
    return json.loads(Path(path).read_text())


def prepare(root, python, pointwise_plan):
    # This audits all 4,000 Jev calls and verifies each saved selection against
    # the frozen selector before correctness labels are loaded.
    spec = spec_from_file_location("math500_pointwise", root / "scripts/math500_pointwise.py")
    pointwise = module_from_spec(spec)
    spec.loader.exec_module(pointwise)
    jev_audit = pointwise.audit(root, pointwise_plan)
    if jev_audit["status"] != "complete" or jev_audit["selected_answers"] != 500:
        raise ValueError("Require an audited, complete MATH500 pointwise Jev run")

    policy = load(root / scorer.POLICY)
    validation = scorer.validate(root, python)
    validation_manifest = load(validation / "manifest.json")
    controls = load(validation / "controls.json")
    runtime = scorer.worker(root, python, "runtime", policy)
    if (validation_manifest["status"] != "passed" or controls["status"] != "passed"
            or validation_manifest["runtime"] != runtime
            or validation_manifest["implementation_sha256"] != scorer.implementation(root)):
        raise ValueError("Frozen MATH500 scorer validation did not pass")

    archive = root / load(pointwise_plan / "plan.json")["archive"]
    references = root / "data/private/math500-heldout-references.json"
    reference_manifest = load(root / "data/manifests/math500-week01.json")
    if digest(references) != reference_manifest["files"]["data/private/math500-heldout-references.json"]:
        raise ValueError("Pinned MATH500 references differ from the frozen manifest")

    out = selection.new_run(root / "artifacts/runs", "math500-pointwise-scoring-plan")
    plan = {
        "version": policy["version"],
        "pointwise_plan": str(pointwise_plan.resolve()),
        "pointwise_plan_files_sha256": inventory(pointwise_plan),
        "archive": str(archive.resolve()),
        "archive_files_sha256": inventory(archive),
        "references_sha256": digest(references),
        "scorer_policy": policy,
        "scorer_implementation_sha256": scorer.implementation(root),
        "evaluator_sha256": digest(Path(__file__)),
        "runtime": runtime,
        "validation": str(validation.resolve()),
        "validation_files_sha256": inventory(validation),
        "question_count": 500,
        "candidate_count": 4000,
        "is_benchmark_result": False,
        "new_api_calls": 0,
        "note": "Offline held-out analysis of the frozen Jev pointwise selector. Math-Verify is an automated scorer; report separately from official benchmark claims.",
    }
    selection.write_json(out / "plan.json", plan)
    return out


def inspect(root, python, plan_dir):
    plan = load(plan_dir / "plan.json")
    if (plan["evaluator_sha256"] != digest(Path(__file__))
            or plan["scorer_implementation_sha256"] != scorer.implementation(root)
            or plan["scorer_policy"] != load(root / scorer.POLICY)):
        raise ValueError("Frozen scoring policy or evaluator changed after prepare")
    if (inventory(Path(plan["pointwise_plan"])) != plan["pointwise_plan_files_sha256"]
            or inventory(Path(plan["archive"])) != plan["archive_files_sha256"]
            or inventory(Path(plan["validation"])) != plan["validation_files_sha256"]
            or digest(root / "data/private/math500-heldout-references.json") != plan["references_sha256"]
            or scorer.worker(root, python, "runtime", plan["scorer_policy"]) != plan["runtime"]):
        raise ValueError("Frozen scoring inputs, runtime, or validation artifacts changed")
    module_spec = spec_from_file_location("math500_pointwise", root / "scripts/math500_pointwise.py")
    pointwise = module_from_spec(module_spec)
    module_spec.loader.exec_module(pointwise)
    audit = pointwise.audit(root, Path(plan["pointwise_plan"]))
    if audit["status"] != "complete" or audit["selected_answers"] != plan["question_count"]:
        raise ValueError("Pointwise Jev output no longer passes its audit")
    archive = Path(plan["archive"])
    questions, pools = math500_jev.public_inputs(root, archive)
    if len(questions) != 500 or sum(len(pool) for pool in pools.values()) != 4000:
        raise ValueError("Expected the complete frozen 500 by 8 MATH500 pool")
    pointwise_selections = [json.loads(line) for line in
        (Path(plan["pointwise_plan"]) / "live/selections.jsonl").read_text().splitlines() if line.strip()]
    pointwise_scores = [json.loads(line) for line in
        (Path(plan["pointwise_plan"]) / "live/candidate-scores.jsonl").read_text().splitlines() if line.strip()]
    if len(pointwise_selections) != 500:
        raise ValueError("Jev selection file is incomplete")
    if len(pointwise_scores) != 4000:
        raise ValueError("Jev candidate score file is incomplete")
    return plan, questions, pools, pointwise_selections, pointwise_scores


def evaluate(root, python, plan_dir):
    plan, questions, pools, pointwise_selections, pointwise_scores = inspect(root, python, plan_dir)
    policy = plan["scorer_policy"]
    refs = load(root / "data/private/math500-heldout-references.json")
    references = {row["question_id"]: row["answer"] for row in refs}
    if len(references) != 500 or set(references) != {q.question_id for q in questions}:
        raise ValueError("Pinned references do not cover the exact MATH500 question set")
    selected = {row["question_id"]: row["candidate_id"] for row in pointwise_selections}
    call_totals = {}
    for row in pointwise_scores:
        totals = call_totals.setdefault(row["question_id"], {"input_tokens": 0, "output_tokens": 0, "request_seconds": 0.0})
        totals["input_tokens"] += row["usage"]["input_tokens"]
        totals["output_tokens"] += row["usage"]["output_tokens"]
        totals["request_seconds"] += row["request_seconds"]
    info = {"questions": questions, "pools": pools,
            "accepted": [{"question_id": q.question_id, "candidate_id": selected[q.question_id],
                          "usage": {k: call_totals[q.question_id][k] for k in ("input_tokens", "output_tokens")},
                          "request_seconds": call_totals[q.question_id]["request_seconds"]} for q in questions],
            "failures": [], "total_attempts": 4000}

    out = selection.new_run(root / "artifacts/runs", "math500-pointwise-evaluation")
    private = out / "evaluation_private"
    private.mkdir()
    os.chmod(out, 0o700)
    os.chmod(private, 0o700)
    manifest = {"status": "running", "plan": str(plan_dir.resolve()),
                "plan_sha256": digest(plan_dir / "plan.json"), "completed_questions": 0,
                "new_api_calls": 0, "new_generation": False, "is_benchmark_result": False}
    selection.write_json(out / "manifest.json", manifest)
    print(f"MATH500 pointwise evaluation directory: {out}", flush=True)
    try:
        reference_controls = scorer.worker(root, python, "reference-controls", policy, references=refs)
        selection.write_json(private / "reference-controls.json", reference_controls)
        if reference_controls["status"] != "passed":
            raise ValueError("Pinned reference controls failed")

        def score(question):
            candidates = [{key: asdict(c)[key] for key in
                           ("candidate_id", "text", "completed", "truncated")}
                          for c in pools[question.question_id]]
            return scorer.worker(root, python, "question", policy, question={
                "question_id": question.question_id, "candidates": candidates,
                "reference": references[question.question_id]})

        records = []
        with ThreadPoolExecutor(max_workers=policy["workers"]) as executor, \
                (private / "question-scores.jsonl").open("x") as stream:
            for record in executor.map(score, questions):
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                manifest["completed_questions"] = len(records)
                selection.write_json(out / "manifest.json", manifest)
                if len(records) % 10 == 0 or len(records) == 500:
                    print(f"MATH500 scoring: {len(records)}/500 questions", flush=True)

        metrics, details = scorer.summarize(info, records, policy)
        metrics["analysis_type"] = "frozen_pointwise_selection_with_post_selection_math_verify_scoring"
        metrics["jev_execution"] = {
            "selector": "pointwise", "successful_candidate_calls": len(pointwise_scores),
            "selected_answers": len(pointwise_selections), "total_attempted_calls": len(pointwise_scores),
            "rejected_responses": 0, "automatic_retries": 0, "new_api_calls_during_scoring": 0,
            "usage": {key: sum(row["usage"][key] for row in pointwise_scores)
                      for key in ("input_tokens", "output_tokens")},
            "request_seconds_sum": sum(row["request_seconds"] for row in pointwise_scores),
            "api_cost_usd": None,
        }
        # Recheck immutable input evidence immediately before writing results.
        inspect(root, python, plan_dir)
        selection.write_json(out / "metrics.json", metrics)
        selection.write_json(private / "question-results.json", details)
        summary = ["# MATH500 Held-Out Jev Pointwise Evaluation", "",
            "This is an offline evaluation of the frozen Jev pointwise selector on the 500-question held-out MATH500 pool.",
            "The pool contains eight generated candidates per question. No GPU or new generation was used.", "",
            "| Selector | Accuracy | Correct questions |", "|---|---:|---:|"]
        for name in ("jev", "first_sample", "seeded_random", "majority_vote"):
            summary.append(f"| {name} | {metrics['accuracy'][name]:.1%} | {metrics['correct_counts'][name]}/500 |")
        summary += ["", f"Empirical oracle@8: {metrics['empirical_oracle_at_pool_size']:.1%}.",
            "", "**Interpretation:** Results use the project's pinned Math-Verify scorer and are marked `is_benchmark_result: false`; treat them as the project's held-out evaluation under this scorer, not as an official MATH benchmark score.", ""]
        (out / "summary.md").write_text("\n".join(summary))
        manifest.update(status="complete", outputs_sha256={
            str(f.relative_to(out)): digest(f) for f in sorted(out.rglob("*"))
            if f.is_file() and f.name != "manifest.json"})
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__)
        raise
    finally:
        selection.write_json(out / "manifest.json", manifest)
    return out


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "evaluate"))
    parser.add_argument("--pointwise-plan", type=Path)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--math-python", type=Path, default=Path(".venv-math/bin/python"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Keep the venv symlink path intact; resolving it can bypass the venv on macOS.
    python = (root / args.math_python).absolute() if not args.math_python.is_absolute() else args.math_python
    try:
        if args.action == "prepare":
            if not args.pointwise_plan:
                parser.error("prepare requires --pointwise-plan")
            out = prepare(root, python, (root / args.pointwise_plan).resolve())
            result = {"status": "prepared", "plan_directory": str(out),
                      "new_api_calls": 0, "heldout_answers_scored": False}
        else:
            if not args.plan_dir:
                parser.error(f"{args.action} requires --plan-dir")
            plan_dir = (root / args.plan_dir).resolve()
            if args.action == "check":
                p, *_ = inspect(root, python, plan_dir)
                result = {"status": "passed", "questions": p["question_count"],
                          "candidates": p["candidate_count"], "new_api_calls": 0}
            else:
                result = {"status": "complete", "output": str(evaluate(root, python, plan_dir))}
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped; saved generation and Jev evidence are unchanged.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"MATH500 pointwise scoring stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
