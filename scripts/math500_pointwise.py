"""Apply the frozen Week 4 pointwise Jev selector to the held-out MATH500 pool."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_dllm.experiments import frozen
from jev_dllm.experiments import math500_jev, selection, week2_jev
from jev_dllm.judges.credentials import load_api_key
from jev_dllm.judges.pointwise import PointwiseJudge, build_pointwise_request, parse_pointwise_response
from jev_dllm.reproducibility import fingerprint
from jev_dllm.selection import frozen_selector


QUESTIONS = "data/processed/math500-heldout.json"
EXPECTED_QUESTIONS = 500
EXPECTED_CANDIDATES = 4000
BENCHMARK = "math500"


def sha256(path: Path) -> str:
    return frozen.sha256(path)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL record: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def install_audited_recovery(root: Path, archive: Path) -> None:
    """Load the exact recovery source pinned by this completed recovery plan.

    Recovery tooling may evolve after a run. Keep its original audit meaningful
    by checking the archived implementation bytes and every companion file
    against the saved plan, rather than refreshing any recorded hash.
    """
    recovery_path = "src/jev_dllm/experiments/collection_recovery.py"
    recovery_plan = json.loads((archive / "recovery-plan.json").read_text())
    expected = recovery_plan.get("extension_sha256", {})
    if recovery_path not in expected:
        raise ValueError("Recovery plan does not pin the collection recovery source")
    evidence = root / "artifacts/recovery-evidence" / f"collection_recovery-{expected[recovery_path][:12]}.py"
    if sha256(evidence) != expected[recovery_path]:
        raise ValueError("The exact recovery implementation pinned by the audited plan is unavailable")
    for name, pinned_hash in expected.items():
        if name != recovery_path and sha256(root / name) != pinned_hash:
            raise ValueError(f"Recovery companion file differs from the audited plan: {name}")

    module_name = "jev_dllm.experiments._audited_collection_recovery"
    spec = importlib.util.spec_from_file_location(module_name, evidence)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load the audited recovery implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def pinned_extension_hashes(check_root):
        if sha256(evidence) != expected[recovery_path]:
            raise ValueError("Archived recovery implementation changed")
        current = {name: sha256(check_root / name) for name in expected if name != recovery_path}
        if any(current[name] != expected[name] for name in current):
            raise ValueError("Recovery companion implementation changed")
        return dict(expected)

    module.extension_hashes = pinned_extension_hashes
    math500_jev.collection_recovery = module


def inputs(root: Path, archive: Path):
    install_audited_recovery(root, archive)
    selector_record = frozen_selector.check(root)
    questions, pools = math500_jev.public_inputs(root, archive)
    if len(questions) != EXPECTED_QUESTIONS or sum(map(len, pools.values())) != EXPECTED_CANDIDATES:
        raise ValueError("Require the complete 500-question, eight-candidate MATH500 pool")
    if any(q.dataset != "math500" or q.split != "heldout" for q in questions):
        raise ValueError("Expected the pinned held-out MATH500 question set")
    protocol = frozen_selector.prompts(BENCHMARK)
    prompt_text = {key: (root / path).read_text() for key, path in protocol.items()}
    prompts_sha = {key: sha256(root / path) for key, path in protocol.items()}
    expected_prompts = {key: selector_record["files_sha256"][path] for key, path in protocol.items()}
    if prompts_sha != expected_prompts:
        raise ValueError("MATH500 pointwise prompts differ from the frozen selector record")
    return selector_record, questions, pools, protocol, prompt_text, prompts_sha


def make_requests(questions, pools, prompt_text, model):
    rows = []
    for question in questions:
        for candidate in pools[question.question_id]:
            rows.append({
                "call_index": len(rows),
                "question_id": question.question_id,
                "candidate_id": candidate.candidate_id,
                "body": build_pointwise_request(
                    question, candidate, prompt_text["correct"], prompt_text["consistent"], model),
            })
    if len(rows) != EXPECTED_CANDIDATES:
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} pointwise requests, got {len(rows)}")
    return rows


def prepare(root: Path, archive: Path) -> Path:
    record, questions, pools, prompt_paths, prompt_text, prompt_hashes = inputs(root, archive)
    model = record["jev"]["model"]
    requests = make_requests(questions, pools, prompt_text, model)
    out = selection.new_run(root / "artifacts/runs", "math500-pointwise-plan")
    requests_path = out / "requests.jsonl"
    requests_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests))
    receipt = math500_jev.collection_recovery.receipt_for(archive)
    plan = {
        "version": "week4_frozen_math500_pointwise_v1",
        "mode": "frozen_pointwise_math500",
        "status": "prepared",
        "task": "math500",
        "archive": archive.relative_to(root).as_posix(),
        "run_id": archive.name,
        "source_candidates_sha256": sha256(archive / "candidates.jsonl"),
        "source_snapshot_digest": math500_jev.snapshot_digest(receipt),
        "questions_sha256": sha256(root / QUESTIONS),
        "protocol": json.loads((archive / "manifest.json").read_text())["protocol"],
        "selector_record": frozen_selector.RECORD,
        "selector_record_sha256": sha256(root / frozen_selector.RECORD),
        "selector": record["selector"],
        "model": model,
        "jev_protocol": record["jev"]["protocol"],
        "prompt_paths": prompt_paths,
        "prompt_sha256": prompt_hashes,
        "request_count": len(requests),
        "requests_sha256": sha256(requests_path),
        "implementation_sha256": sha256(Path(__file__)),
        "paid_calls_sent": 0,
        "planned_calls": {"pointwise": len(requests), "total": len(requests)},
        "note": "One pointwise Jev request per candidate; select highest correct probability, ties by original pool order.",
    }
    write_json(out / "plan.json", plan)
    return out


def check(root: Path, plan_dir: Path):
    plan = json.loads((plan_dir / "plan.json").read_text())
    if plan.get("version") != "week4_frozen_math500_pointwise_v1" or plan.get("task") != "math500":
        raise ValueError("Unknown MATH500 pointwise plan")
    archive = root / plan["archive"]
    record, questions, pools, prompt_paths, prompt_text, prompt_hashes = inputs(root, archive)
    requests = make_requests(questions, pools, prompt_text, record["jev"]["model"])
    saved = read_jsonl(plan_dir / "requests.jsonl")
    receipt = math500_jev.collection_recovery.receipt_for(archive)
    if (plan.get("run_id") != archive.name
            or plan.get("source_candidates_sha256") != sha256(archive / "candidates.jsonl")
            or plan.get("source_snapshot_digest") != math500_jev.snapshot_digest(receipt)
            or plan.get("questions_sha256") != sha256(root / QUESTIONS)
            or plan.get("protocol") != json.loads((archive / "manifest.json").read_text())["protocol"]
            or plan.get("selector_record_sha256") != sha256(root / frozen_selector.RECORD)
            or plan.get("selector") != record["selector"]
            or plan.get("model") != record["jev"]["model"]
            or plan.get("jev_protocol") != record["jev"]["protocol"]
            or plan.get("prompt_paths") != prompt_paths
            or plan.get("prompt_sha256") != prompt_hashes
            or plan.get("implementation_sha256") != sha256(Path(__file__))
            or plan.get("request_count") != EXPECTED_CANDIDATES
            or plan.get("planned_calls") != {"pointwise": EXPECTED_CANDIDATES, "total": EXPECTED_CANDIDATES}
            or plan.get("paid_calls_sent") != 0
            or plan.get("requests_sha256") != sha256(plan_dir / "requests.jsonl")
            or saved != requests):
        raise ValueError("Plan or public inputs differ from the frozen MATH500 pointwise protocol")
    return plan, requests, questions, pools


def live(root: Path, plan_dir: Path, approved_calls: int, env_file: Path) -> Path:
    plan, requests, questions, pools = check(root, plan_dir)
    if approved_calls != EXPECTED_CANDIDATES:
        raise ValueError(f"APPROVED_CALLS must equal {EXPECTED_CANDIDATES}")
    load_api_key(env_file)
    claims = root / "artifacts/math500-pointwise-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_path = claims / f"{plan['run_id']}-{plan['requests_sha256']}.json"
    with claim_path.open("x") as stream:
        json.dump({"plan": str(plan_dir.relative_to(root)), "requests_sha256": plan["requests_sha256"]}, stream)
        stream.write("\n")
    out = plan_dir / "live"
    out.mkdir(exist_ok=False)
    manifest = {
        "mode": "live_jev", "status": "running", "task": "math500",
        "plan_sha256": sha256(plan_dir / "plan.json"), "run_id": plan["run_id"],
        "request_sha256": plan["requests_sha256"], "model": plan["model"],
        "successful_calls": 0, "attempted_calls": 0, "expected_calls": EXPECTED_CANDIDATES,
        "questions_sha256": fingerprint([asdict(q) for q in questions]),
        "pools_sha256": fingerprint({k: [asdict(c) for c in v] for k, v in pools.items()}),
        "protocol": plan["protocol"], "api_cost_usd": None,
        "retry_policy": "none; preserve accepted prefix and stop on the first error",
    }
    write_json(out / "manifest.json", manifest)
    judge = PointwiseJudge()
    result_path = out / "candidate-scores.jsonl"
    try:
        with result_path.open("x") as stream:
            for row in requests:
                manifest["attempted_calls"] += 1
                write_json(out / "manifest.json", manifest)
                try:
                    raw = judge.judge(row["body"])
                    parsed = parse_pointwise_response(raw["raw_response"], row["body"])
                except Exception as exc:
                    if hasattr(exc, "raw_response"):
                        write_json(out / "rejected_response.json", {
                            "call_index": row["call_index"], "question_id": row["question_id"],
                            "candidate_id": row["candidate_id"], "raw_response": exc.raw_response,
                            "request_seconds": getattr(exc, "request_seconds", None),
                        })
                    manifest.update(status="failed", error_type=type(exc).__name__,
                                    failed_call_index=row["call_index"], failed_candidate_id=row["candidate_id"])
                    write_json(out / "manifest.json", manifest)
                    raise
                record = {k: row[k] for k in ("call_index", "question_id", "candidate_id")}
                record.update({k: parsed[k] for k in ("correct_probability", "consistent_probability", "model", "usage")})
                record["request_seconds"] = raw["request_seconds"]
                record["raw_response"] = raw["raw_response"]
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                manifest["successful_calls"] += 1
                write_json(out / "manifest.json", manifest)
                if manifest["successful_calls"] % 100 == 0:
                    print(f"MATH500 Jev pointwise: {manifest['successful_calls']}/{EXPECTED_CANDIDATES}", flush=True)
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error_type="KeyboardInterrupt")
        write_json(out / "manifest.json", manifest)
        raise

    scores = read_jsonl(result_path)
    score_map = {r["candidate_id"]: r["correct_probability"] for r in scores}
    if len(scores) != EXPECTED_CANDIDATES or len(score_map) != EXPECTED_CANDIDATES:
        raise ValueError("Pointwise results do not contain one score per candidate")
    selected = []
    for question in questions:
        ids = [c.candidate_id for c in pools[question.question_id]]
        candidate_id = frozen_selector.select(ids, score_map)
        selected.append({"question_id": question.question_id, "candidate_id": candidate_id,
                         "correct_probability": score_map[candidate_id]})
    (out / "selections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    manifest.update(status="complete", selected_questions=len(selected),
                    selection_rule="highest_correct_probability_ties_by_pool_order")
    write_json(out / "manifest.json", manifest)
    return out


def audit(root: Path, plan_dir: Path):
    plan, requests, questions, pools = check(root, plan_dir)
    out = plan_dir / "live"
    manifest = json.loads((out / "manifest.json").read_text())
    scores = read_jsonl(out / "candidate-scores.jsonl")
    selections = read_jsonl(out / "selections.jsonl")
    if (manifest.get("status") != "complete" or manifest.get("successful_calls") != EXPECTED_CANDIDATES
            or manifest.get("attempted_calls") != EXPECTED_CANDIDATES or len(scores) != EXPECTED_CANDIDATES
            or len(selections) != EXPECTED_QUESTIONS):
        raise ValueError("Require a complete 4,000-call pointwise run before auditing")
    score_map = {}
    for row, request in zip(scores, requests):
        parsed = parse_pointwise_response(row["raw_response"], request["body"])
        if (row.get("candidate_id") != request["candidate_id"]
                or row.get("question_id") != request["question_id"]
                or parsed["correct_probability"] != row.get("correct_probability")):
            raise ValueError("A saved pointwise response differs from its request")
        score_map[row["candidate_id"]] = row["correct_probability"]
    for q, row in zip(questions, selections):
        ids = [c.candidate_id for c in pools[q.question_id]]
        expected = frozen_selector.select(ids, score_map)
        if row.get("question_id") != q.question_id or row.get("candidate_id") != expected:
            raise ValueError("Saved selection differs from the frozen selector")
    summary = {"status": "complete", "task": "math500", "questions": len(questions),
               "candidates": len(scores), "pointwise_calls": len(scores),
               "selected_answers": len(selections), "selector": "jev_pointwise",
               "correctness_scored": False, "api_cost_usd": None,
               "input_tokens": sum(r["usage"]["input_tokens"] for r in scores),
               "output_tokens": sum(r["usage"]["output_tokens"] for r in scores),
               "request_seconds": sum(r["request_seconds"] for r in scores)}
    write_json(out / "audit-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "live", "audit"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--approved-calls", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "prepare":
            if not args.archive:
                parser.error("prepare requires --archive")
            out = prepare(root, (root / args.archive).resolve())
            plan = json.loads((out / "plan.json").read_text())
            result = {"status": "prepared", "plan_directory": str(out),
                      "planned_calls": plan["planned_calls"], "paid_calls_sent": 0}
        elif args.action == "check":
            if not args.plan_dir:
                parser.error("check requires --plan-dir")
            plan, requests, *_ = check(root, (root / args.plan_dir).resolve())
            result = {"status": "passed", "requests": len(requests),
                      "planned_calls": plan["planned_calls"], "paid_calls_sent": 0}
        elif args.action == "live":
            if not args.plan_dir or args.approved_calls is None:
                parser.error("live requires --plan-dir and --approved-calls")
            out = live(root, (root / args.plan_dir).resolve(), args.approved_calls, root / args.env_file)
            result = {"status": "complete", "live_directory": str(out)}
        else:
            if not args.plan_dir:
                parser.error("audit requires --plan-dir")
            result = audit(root, (root / args.plan_dir).resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped; retain the partial pointwise evidence and do not relaunch blindly.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"MATH500 pointwise stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
