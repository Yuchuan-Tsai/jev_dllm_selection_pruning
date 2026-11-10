"""Frozen GSM8K-500 pointwise Jev evaluation.

This workflow keeps the Week 4 selector (one pointwise score per candidate,
then highest `correct`) separate from the older GSM8K listwise judge target.
Preparation and checking are offline. Only `live` sends paid Jev requests.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_dllm.config import load
from jev_dllm.experiments import frozen, selection, week2_jev
from jev_dllm.judges.credentials import load_api_key
from jev_dllm.judges.pointwise import PointwiseJudge, build_pointwise_request
from jev_dllm.reproducibility import fingerprint
from jev_dllm.selection import frozen_selector


PROFILE = "configs/week01/gsm8k500.json"
QUESTIONS = "data/processed/gsm8k-heldout.json"
REFERENCES = "data/private/gsm8k-heldout-references.json"
TASK = "gsm8k500"
EXPECTED_CALLS = 4000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL record in {path}")
    return [json.loads(line) for line in raw.splitlines()]


def load_pool(root: Path, run_dir: Path):
    config, pending = load(root / PROFILE)
    if pending:
        raise ValueError("Frozen GSM8K profile has unresolved fields")
    selector_record = frozen_selector.check(root)
    protocol = frozen.check_pool(root, config, root / QUESTIONS, run_dir / "candidates.jsonl")
    questions, pools = selection.load_pools(root / QUESTIONS, run_dir / "candidates.jsonl")
    if len(questions) != 500 or sum(len(pool) for pool in pools.values()) != EXPECTED_CALLS:
        raise ValueError("Expected 500 held-out GSM8K questions with eight candidates each")
    if any(q.split != "heldout" or q.dataset != "gsm8k" for q in questions):
        raise ValueError("Pointwise held-out evaluation only accepts the frozen GSM8K-500 input")
    return config, selector_record, protocol, questions, pools


def build_requests(root: Path, questions, pools, model: str) -> list[dict]:
    prompts = week2_jev.prompt_texts(root, "gsm8k")
    rows = []
    for question in questions:
        for candidate in pools[question.question_id]:
            rows.append({
                "call_index": len(rows),
                "question_id": question.question_id,
                "candidate_id": candidate.candidate_id,
                "body": build_pointwise_request(
                    question, candidate, prompts["correct"], prompts["consistent"],
                    model,
                ),
            })
    if len(rows) != EXPECTED_CALLS:
        raise ValueError(f"Expected {EXPECTED_CALLS} pointwise requests, got {len(rows)}")
    return rows


def prepare(root: Path, run_dir: Path) -> Path:
    config, selector_record, protocol, questions, pools = load_pool(root, run_dir)
    requests = build_requests(root, questions, pools, selector_record["jev"]["model"])
    out = selection.new_run(root / "artifacts/runs", "gsm8k500-pointwise-plan")
    request_path = out / "requests.jsonl"
    request_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests))
    plan = {
        "schema_version": 1,
        "mode": "frozen_pointwise_gsm8k500",
        "status": "prepared",
        "run_id": run_dir.name,
        "pool_path": run_dir.relative_to(root).as_posix(),
        "pool_sha256": digest(run_dir / "candidates.jsonl"),
        "manifest_sha256": digest(run_dir / "manifest.json"),
        "questions_sha256": digest(root / QUESTIONS),
        "protocol": protocol,
        "selector_record": frozen_selector.RECORD,
        "selector_record_sha256": digest(root / frozen_selector.RECORD),
        "selector": selector_record["selector"],
        "model": selector_record["jev"]["model"],
        "jev_protocol": selector_record["jev"]["protocol"],
        "request_count": len(requests),
        "requests_sha256": digest(request_path),
        "request_builder_sha256": digest(Path(__file__)),
        "planned_calls": {"pointwise": len(requests), "total": len(requests)},
        "paid_calls_sent": 0,
        "references_used_during_preparation": False,
        "note": "One Jev pointwise request per candidate; selection uses highest correct_probability, ties by pool order.",
    }
    write_json(out / "plan.json", plan)
    return out


def check_plan(root: Path, plan_dir: Path):
    plan = json.loads((plan_dir / "plan.json").read_text())
    if plan.get("schema_version") != 1 or plan.get("mode") != "frozen_pointwise_gsm8k500":
        raise ValueError("Unknown GSM8K-500 pointwise plan")
    run_dir = root / plan["pool_path"]
    config, selector_record, protocol, questions, pools = load_pool(root, run_dir)
    requests = build_requests(root, questions, pools, selector_record["jev"]["model"])
    request_path = plan_dir / "requests.jsonl"
    saved_requests = read_jsonl(request_path)
    if (plan.get("run_id") != run_dir.name
            or plan.get("pool_sha256") != digest(run_dir / "candidates.jsonl")
            or plan.get("manifest_sha256") != digest(run_dir / "manifest.json")
            or plan.get("questions_sha256") != digest(root / QUESTIONS)
            or plan.get("protocol") != protocol
            or plan.get("selector_record_sha256") != digest(root / frozen_selector.RECORD)
            or plan.get("selector") != selector_record["selector"]
            or plan.get("model") != selector_record["jev"]["model"]
            or plan.get("jev_protocol") != selector_record["jev"]["protocol"]
            or plan.get("request_builder_sha256") != digest(Path(__file__))
            or plan.get("requests_sha256") != digest(request_path)
            or plan.get("request_count") != EXPECTED_CALLS
            or plan.get("planned_calls") != {"pointwise": EXPECTED_CALLS, "total": EXPECTED_CALLS}
            or plan.get("paid_calls_sent") != 0
            or saved_requests != requests):
        raise ValueError("Plan, requests, pool, selector freeze, or implementation changed")
    return plan, requests, config, questions, pools


def live(root: Path, plan_dir: Path, approved_calls: int, env_file: Path):
    plan, requests, config, questions, pools = check_plan(root, plan_dir)
    if approved_calls != EXPECTED_CALLS:
        raise ValueError(f"APPROVED_CALLS must equal {EXPECTED_CALLS}")
    load_api_key(env_file)
    claims = root / "artifacts/gsm8k500-pointwise-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_path = claims / f"{plan['run_id']}-{plan['requests_sha256']}.json"
    with claim_path.open("x") as stream:
        json.dump({"plan": str(plan_dir.relative_to(root)), "requests_sha256": plan["requests_sha256"]}, stream)
        stream.write("\n")

    out = selection.new_run(root / "artifacts/runs", "gsm8k500-pointwise-live")
    results_path = out / "candidate-scores.jsonl"
    manifest = {
        "mode": "live_jev", "status": "running", "task": TASK,
        "run_id": plan["run_id"], "plan_path": str(plan_dir.relative_to(root)),
        "plan_sha256": digest(plan_dir / "plan.json"),
        "request_sha256": plan["requests_sha256"], "model": plan["model"],
        "master_seed": config["runtime"]["master_seed"],
        "expected_calls": EXPECTED_CALLS, "attempted_calls": 0,
        "successful_calls": 0, "questions_sha256": fingerprint([asdict(q) for q in questions]),
        "pools_sha256": fingerprint({k: [asdict(c) for c in v] for k, v in pools.items()}),
        "protocol": plan["protocol"], "api_cost_usd": None,
        "retry_policy": "none; stop and preserve accepted prefix on first failure",
    }
    write_json(out / "manifest.json", manifest)
    judge = PointwiseJudge()
    try:
        with results_path.open("x") as stream:
            for row in requests:
                manifest["attempted_calls"] += 1
                write_json(out / "manifest.json", manifest)
                try:
                    result = judge.judge(row["body"])
                except Exception as exc:
                    if hasattr(exc, "raw_response"):
                        write_json(out / "rejected_response.json", {
                            "call_index": row["call_index"],
                            "question_id": row["question_id"],
                            "candidate_id": row["candidate_id"],
                            "raw_response": exc.raw_response,
                            "request_seconds": getattr(exc, "request_seconds", None),
                        })
                    manifest.update(status="failed", error_type=type(exc).__name__,
                                    failed_call_index=row["call_index"],
                                    failed_candidate_id=row["candidate_id"])
                    write_json(out / "manifest.json", manifest)
                    raise
                record = {k: row[k] for k in ("call_index", "question_id", "candidate_id")}
                record.update({k: result[k] for k in (
                    "correct_probability", "consistent_probability", "model", "usage", "request_seconds")})
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                manifest["successful_calls"] += 1
                write_json(out / "manifest.json", manifest)
                if manifest["successful_calls"] % 100 == 0:
                    print(f"Jev pointwise: {manifest['successful_calls']}/{EXPECTED_CALLS} calls", flush=True)
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error_type="KeyboardInterrupt")
        write_json(out / "manifest.json", manifest)
        raise
    except BaseException:
        raise

    scores = read_jsonl(results_path)
    score_map = {row["candidate_id"]: row["correct_probability"] for row in scores}
    if len(score_map) != EXPECTED_CALLS:
        raise ValueError("Jev score output is incomplete or contains duplicate candidate IDs")
    selection_rows = []
    for question in questions:
        ids = [c.candidate_id for c in pools[question.question_id]]
        selected_id = frozen_selector.select(ids, score_map)
        selection_rows.append({
            "question_id": question.question_id,
            "candidate_id": selected_id,
            "correct_probability": score_map[selected_id],
            "selector": "highest_correct_probability_ties_by_pool_order",
        })
    (out / "selections.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows))
    manifest.update(status="complete", successful_calls=EXPECTED_CALLS,
                    selected_questions=len(selection_rows), selection_rule=plan["selector"])
    write_json(out / "manifest.json", manifest)
    return out


def evaluate(root: Path, live_dir: Path):
    manifest = json.loads((live_dir / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("successful_calls") != EXPECTED_CALLS
            or manifest.get("expected_calls") != EXPECTED_CALLS):
        raise ValueError("Require a complete 4,000-call pointwise run")
    plan_dir = root / manifest["plan_path"]
    plan, requests, config, questions, pools = check_plan(root, plan_dir)
    run_dir = root / plan["pool_path"]
    if (manifest.get("run_id") != run_dir.name
            or manifest.get("protocol") != plan["protocol"]
            or manifest.get("request_sha256") != plan["requests_sha256"]):
        raise ValueError("Live results do not match the verified GSM8K-500 plan")
    score_rows = read_jsonl(live_dir / "candidate-scores.jsonl")
    request_ids = {(r["question_id"], r["candidate_id"]) for r in requests}
    score_ids = {(r["question_id"], r["candidate_id"]) for r in score_rows}
    if len(score_rows) != EXPECTED_CALLS or score_ids != request_ids:
        raise ValueError("Pointwise scores do not cover every planned candidate exactly once")
    if digest(root / REFERENCES) != config["evaluation"]["references_sha256"]:
        raise ValueError("Private GSM8K references differ from the frozen pin")
    if config["evaluation"]["scorer"] != "gsm8k_units_v3":
        raise ValueError("Unexpected frozen GSM8K scorer")
    from jev_dllm.experiments.selection import evaluate_pools
    out = evaluate_pools(root / QUESTIONS, run_dir / "candidates.jsonl", root / REFERENCES,
                         root / "artifacts/runs", live_dir / "selections.jsonl",
                         seed=config["runtime"]["master_seed"], scorer="gsm8k_units_v3",
                         protocol=plan["protocol"])
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["is_benchmark_result"] = True
    metrics["selector"] = "frozen_week4_jev_pointwise"
    metrics["selection_run_id"] = live_dir.name
    metrics["selection_calls"] = EXPECTED_CALLS
    write_json(out / "metrics.json", metrics)
    (out / "summary.md").write_text(
        "# GSM8K-500 held-out evaluation\n\n"
        f"Questions: {metrics['question_count']}  \n"
        f"Selector: frozen Jev pointwise (highest `correct` probability)  \n"
        f"Jev calls: {EXPECTED_CALLS}\n\n"
        "| Method | Accuracy |\n|---|---:|\n" +
        "".join(f"| {name} | {value:.1%} |\n" for name, value in metrics["accuracy"].items()) +
        "\nPaired bootstrap intervals and question level outcomes are in `metrics.json` and `evaluation_private/`.\n"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "live", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--live-dir", type=Path)
    parser.add_argument("--approved-calls", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "prepare":
            if not args.run_dir:
                parser.error("prepare requires --run-dir")
            out = prepare(root, (root / args.run_dir).resolve())
            plan = json.loads((out / "plan.json").read_text())
            result = {"status": "prepared", "plan_directory": str(out),
                      "planned_calls": plan["planned_calls"], "paid_calls_sent": 0}
        elif args.action == "check":
            if not args.plan_dir:
                parser.error("check requires --plan-dir")
            plan, requests, *_ = check_plan(root, (root / args.plan_dir).resolve())
            result = {"status": "passed", "requests": len(requests),
                      "planned_calls": plan["planned_calls"], "paid_calls_sent": 0}
        elif args.action == "live":
            if not args.plan_dir or args.approved_calls is None:
                parser.error("live requires --plan-dir and --approved-calls")
            out = live(root, (root / args.plan_dir).resolve(), args.approved_calls,
                       (root / args.env_file).resolve())
            result = {"status": "complete", "live_directory": str(out)}
        else:
            if not args.live_dir:
                parser.error("evaluate requires --live-dir")
            out = evaluate(root, (root / args.live_dir).resolve())
            result = {"status": "complete", "evaluation_directory": str(out),
                      "metrics": json.loads((out / "metrics.json").read_text())["accuracy"]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"GSM8K-500 pointwise evaluation stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
