"""Complete Qwen MATH pointwise evaluation while bypassing failed listwise calls."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_dllm.evaluation import week2_labels
from jev_dllm.experiments import frozen, selection, week2_jev
from jev_dllm.judges.credentials import load_api_key
from jev_dllm.judges.pointwise import PointwiseJudge, parse_pointwise_response
from jev_dllm.reproducibility import fingerprint
from jev_dllm.schema import Candidate
from jev_dllm.selection import week3 as selectors
from jev_dllm.selection.subsets import SIZES, nested_subsets
from jev_dllm.evaluation.week3_metrics import compare
from jev_dllm.backends import qwen3_policy
from jev_dllm.experiments.selection import load_questions
from jev_dllm.selection.frozen_selector import select as frozen_select

SOURCE_PLAN = "artifacts/runs/week2-jev-math-plan-20261108T173552Z-88473d47"
EXPECTED_TOTAL = 3200
VERSION = "week4_qwen_math_pointwise_recovery_v1"


def digest(path):
    return frozen.sha256(path)


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_rows(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL evidence: {path}")
    return [json.loads(x) for x in raw.splitlines()]


def source_data(root, source_plan):
    plan, rows = week2_jev.check(root, source_plan)
    if plan.get("task") != "math" or plan.get("pool_dir") != "artifacts/downloads/autodl/w4-qwen3-14b-math-20261107T073720Z-24cc4444":
        raise ValueError("Expected the prepared Qwen MATH Jev plan and its exact candidate pool")
    results = []
    for name in ("live/results.jsonl", "live-continued/results.jsonl"):
        results.extend(read_rows(source_plan / name))
    requests = {r["call_index"]: r for r in rows}
    accepted = {}
    for record in results:
        index = record.get("call_index")
        req = requests.get(index)
        if req is None:
            raise ValueError("Saved Jev evidence contains a call outside the frozen plan")
        if record.get("kind") != "pointwise":
            continue
        parsed = week2_jev.validate_result(record, req)
        cid = req["candidate_id"]
        if parsed.get("candidate_id") != cid or record.get("candidate_id") != cid:
            raise ValueError("Saved pointwise response does not match its request")
        if cid in accepted and accepted[cid] != record:
            raise ValueError("The source and continuation disagree on an accepted pointwise score")
        accepted[cid] = record
    pointwise = [r for r in rows if r["kind"] == "pointwise"]
    if len(pointwise) != EXPECTED_TOTAL or len({r["candidate_id"] for r in pointwise}) != EXPECTED_TOTAL:
        raise ValueError("Frozen MATH plan does not cover exactly 3,200 pointwise candidates")
    remaining = [r for r in pointwise if r["candidate_id"] not in accepted]
    return plan, pointwise, accepted, remaining


def prepare(root, source_plan):
    plan, pointwise, accepted, remaining = source_data(root, source_plan)
    out = selection.new_run(root / "artifacts/runs", "qwen-math-pointwise-recovery-plan")
    (out / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in remaining))
    (out / "reused-scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted.values()))
    manifest = {"version": VERSION, "task": "math", "source_plan": source_plan.relative_to(root).as_posix(),
                "source_plan_sha256": digest(source_plan / "plan.json"),
                "source_requests_sha256": digest(source_plan / "requests.jsonl"),
                "source_evidence_sha256": {p: digest(source_plan / p) for p in
                    ("live/manifest.json", "live/results.jsonl", "live-continued/manifest.json", "live-continued/results.jsonl")
                    if (source_plan / p).exists()},
                "pointwise_requests_sha256": fingerprint(pointwise),
                "reused_scores_sha256": digest(out / "reused-scores.jsonl"),
                "remaining_requests_sha256": digest(out / "requests.jsonl"),
                "reused_pointwise_calls": len(accepted), "remaining_pointwise_calls": len(remaining),
                "skipped_listwise_calls": 800, "paid_calls_sent": 0,
                "note": "Reuses validated accepted pointwise scores; excludes listwise calls because the frozen selector is pointwise."}
    write_json(out / "plan.json", manifest)
    return out, manifest


def check(root, plan_dir):
    p = json.loads((plan_dir / "plan.json").read_text())
    if p.get("version") != VERSION or p.get("task") != "math":
        raise ValueError("Unknown Qwen MATH pointwise recovery plan")
    source_plan = root / p["source_plan"]
    _, pointwise, accepted, remaining = source_data(root, source_plan)
    current_evidence = {name: digest(source_plan / name) for name in
        ("live/manifest.json", "live/results.jsonl", "live-continued/manifest.json", "live-continued/results.jsonl")
        if (source_plan / name).exists()}
    if (p["source_plan_sha256"] != digest(source_plan / "plan.json")
            or p["source_requests_sha256"] != digest(source_plan / "requests.jsonl")
            or p["source_evidence_sha256"] != current_evidence
            or p["pointwise_requests_sha256"] != fingerprint(pointwise)
            or p["reused_pointwise_calls"] != len(accepted)
            or p["remaining_pointwise_calls"] != len(remaining)
            or p["reused_scores_sha256"] != digest(plan_dir / "reused-scores.jsonl")
            or p["remaining_requests_sha256"] != digest(plan_dir / "requests.jsonl")
            or read_rows(plan_dir / "reused-scores.jsonl") != list(accepted.values())
            or read_rows(plan_dir / "requests.jsonl") != remaining):
        raise ValueError("Recovery plan or its source evidence changed")
    return p, remaining


def live(root, plan_dir, approved, env_file):
    p, requests = check(root, plan_dir)
    if type(approved) is not int or approved != len(requests):
        raise ValueError(f"Approve exactly {len(requests)} additional paid pointwise requests")
    load_api_key(env_file)
    claims = root / "artifacts/qwen-math-pointwise-recovery-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = fingerprint({"source_plan": p["source_plan_sha256"], "requests": p["remaining_requests_sha256"]})
    with (claims / f"{claim_id}.json").open("x") as f:
        json.dump({"plan": str(plan_dir.relative_to(root)), "calls": approved}, f)
        f.write("\n")
    out = plan_dir / "live"
    out.mkdir(exist_ok=False)
    manifest = {"status": "running", "approved_calls": approved, "attempted_calls": 0,
                "successful_calls": 0, "automatic_retries": 0, "api_cost_usd": None}
    write_json(out / "manifest.json", manifest)
    judge = PointwiseJudge()
    try:
        with (out / "scores.jsonl").open("x") as stream:
            for req in requests:
                manifest["attempted_calls"] += 1
                write_json(out / "manifest.json", manifest)
                result = judge.judge(req["body"])
                parsed = parse_pointwise_response(result["raw_response"], req["body"])
                if any(result.get(k) != v for k, v in parsed.items()):
                    raise ValueError("Pointwise response disagrees with parsed response")
                saved = {k: req[k] for k in ("call_index", "question_id", "candidate_id")}
                saved.update({k: result[k] for k in ("correct_probability", "consistent_probability", "model", "usage", "request_seconds", "raw_response")})
                stream.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                manifest["successful_calls"] += 1
                write_json(out / "manifest.json", manifest)
                if manifest["successful_calls"] % 100 == 0:
                    print(f"Qwen MATH pointwise recovery: {manifest['successful_calls']}/{approved}", flush=True)
        manifest.update(status="complete", scores_sha256=digest(out / "scores.jsonl"))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__)
        raise
    finally:
        write_json(out / "manifest.json", manifest)
    return out


def analyze(root, plan_dir, math_python):
    p, requests = check(root, plan_dir)
    live_dir = plan_dir / "live"
    m = json.loads((live_dir / "manifest.json").read_text())
    new_scores = read_rows(live_dir / "scores.jsonl")
    old_scores = read_rows(plan_dir / "reused-scores.jsonl")
    if (m.get("status") != "complete" or m.get("successful_calls") != len(requests)
            or m.get("scores_sha256") != digest(live_dir / "scores.jsonl")
            or len(new_scores) != len(requests)):
        raise ValueError("Require all additional pointwise scores to be complete")
    scores = old_scores + new_scores
    plan = json.loads((root / p["source_plan"] / "plan.json").read_text())
    pool_dir = root / plan["pool_dir"]
    archived = qwen3_policy.profile(root, str(pool_dir / "profile.json"))
    pool_manifest = json.loads((pool_dir / "manifest.json").read_text())
    if (pool_manifest.get("status") != "complete" or pool_manifest.get("mode") != "week04_stronger_policy_development"
            or pool_manifest.get("task") != "math" or pool_manifest.get("question_count") != 200
            or pool_manifest.get("completed_candidates") != EXPECTED_TOTAL
            or digest(pool_dir / "profile.json") != pool_manifest.get("profile_sha256")
            or pool_manifest.get("experiment_code_sha256") != {name: digest(root / name) for name in
                ("scripts/week4_qwen.py", "src/jev_dllm/backends/qwen3_policy.py", "src/jev_dllm/reproducibility.py")}
            or pool_manifest.get("files_sha256") != {name: digest(pool_dir / name) for name in
                ("candidates.jsonl", "costs.jsonl", "profile.json")}):
        raise ValueError("Qwen MATH pool, profile, or pinned generation implementation does not verify")
    spec = archived["policy"]["tasks"]["math"]
    questions = load_questions(root / spec["questions"])
    candidates = [Candidate(**r) for r in read_rows(pool_dir / "candidates.jsonl")]
    pools = {q.question_id: [] for q in questions}
    for c in candidates:
        pools[c.question_id].append(c)
    refs = week2_labels.references(root, "math", questions)
    out = selection.new_run(root / "artifacts/runs", "w4-qwen3-analysis-math")
    (out / "evaluation_private").mkdir()
    label_rows, details = week2_labels.math_labels(root, math_python, questions, pools, refs, out)
    correct = {r["candidate_id"]: r["correct"] for r in label_rows}
    score_map = {r["candidate_id"]: r["correct_probability"] for r in scores}
    if len(score_map) != EXPECTED_TOTAL or set(score_map) != set(correct):
        raise ValueError("Combined pointwise scores do not cover every Qwen MATH candidate")
    by_n = {}
    for n in (n for n in SIZES if n <= 16):
        subsets = {q.question_id: nested_subsets([c.candidate_id for c in pools[q.question_id]], (n,))[n]
                   for q in questions}
        picks = {"first": {qid: selectors.first(ids) for qid, ids in subsets.items()},
                 "seeded_random": {qid: selectors.seeded_random(qid, ids, week2_jev.SEED) for qid, ids in subsets.items()},
                 "jev_pointwise": {qid: frozen_select(ids, score_map) for qid, ids in subsets.items()}}
        by_n[str(n)] = compare(picks, subsets, correct, week2_jev.SEED)
    metrics = {"task": "math", "policy": archived["policy"]["model_id"], "questions": len(questions),
               "by_subset_size": by_n, "jev_plan": Path(p["source_plan"]).name,
               "jev_output": plan_dir.name, "pool_run": pool_dir.name,
               "scorer_details": details, "selection_mode": "frozen pointwise; listwise calls skipped",
               "note": "Week 4 development comparison on the Qwen MATH pool; reuses validated pointwise scores and sends no API calls during analysis."}
    write_json(out / "metrics.json", metrics)
    (out / "summary.md").write_text("\n".join(["Task: math; policy: " + metrics["policy"], "",
        "| Selector | " + " | ".join(f"N={n}" for n in by_n) + " |", "|---|" + "---|" * len(by_n)] +
        [f"| {name} | " + " | ".join(f"{by_n[n]['selectors'][name]['accuracy']:.1%}" for n in by_n) + " |"
         for name in ("first", "seeded_random", "jev_pointwise")]) + "\n")
    write_json(out / "manifest.json", {"status": "complete", "task": "math", "pool_run": pool_dir.name,
        "metrics_sha256": digest(out / "metrics.json"), "pointwise_recovery_plan": str(plan_dir), "api_calls": 0})
    (out / "evaluation_private/labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in label_rows))
    return out, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "check", "live", "analyze"))
    ap.add_argument("--source-plan", type=Path, default=Path(SOURCE_PLAN))
    ap.add_argument("--plan-dir", type=Path)
    ap.add_argument("--approved-calls", type=int)
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--math-python", type=Path, default=Path(".venv-math/bin/python"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args(); root = a.root.resolve()
    try:
        source = (root / a.source_plan).resolve()
        if a.action == "prepare":
            out, p = prepare(root, source)
            result = {"status": "prepared", "plan_directory": str(out),
                      "reused_pointwise_calls": p["reused_pointwise_calls"],
                      "additional_paid_calls": p["remaining_pointwise_calls"], "listwise_calls_skipped": 800}
        elif a.action == "check":
            if not a.plan_dir: ap.error("check requires --plan-dir")
            p, req = check(root, (root / a.plan_dir).resolve())
            result = {"status": "passed", "reused_pointwise_calls": p["reused_pointwise_calls"],
                      "additional_paid_calls": len(req), "listwise_calls_skipped": 800}
        elif a.action == "live":
            if not a.plan_dir or a.approved_calls is None: ap.error("live requires --plan-dir and --approved-calls")
            out = live(root, (root / a.plan_dir).resolve(), a.approved_calls, (root / a.env_file).resolve())
            result = {"status": "complete", "live_directory": str(out)}
        else:
            if not a.plan_dir: ap.error("analyze requires --plan-dir")
            math_python = a.math_python if a.math_python.is_absolute() else root / a.math_python
            # Keep the venv executable path intact; resolving its symlink can bypass the venv.
            out, metrics = analyze(root, (root / a.plan_dir).resolve(), math_python)
            n16 = metrics["by_subset_size"]["16"]["selectors"]
            result = {"status": "complete", "analysis_directory": str(out),
                      "n16": {k: n16[k]["accuracy"] for k in ("first", "seeded_random", "jev_pointwise")}}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Qwen MATH pointwise recovery stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
