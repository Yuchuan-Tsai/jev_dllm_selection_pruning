"""Apply the frozen Week 4 Jev pointwise selector to held-out HumanEval."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_dllm.experiments import collection, frozen, humaneval_jev, selection, week2_jev
from jev_dllm.judges.credentials import load_api_key
from jev_dllm.judges.pointwise import PointwiseJudge, build_pointwise_request, parse_pointwise_response
from jev_dllm.selection import frozen_selector
from jev_dllm.reproducibility import fingerprint, trajectory_seed
from jev_dllm.evaluation import humaneval_extracted as extraction
from jev_dllm.selection.baselines import random_choice

BENCHMARK = "humaneval"
QUESTIONS = "data/processed/humaneval-heldout.json"
EXPECTED_QUESTIONS = 164
EXPECTED_CANDIDATES = 1312
VERSION = "week4_frozen_humaneval_pointwise_v1"
FILES = ("scripts/humaneval_pointwise.py", "src/jev_dllm/experiments/humaneval_jev.py",
         "src/jev_dllm/evaluation/humaneval_extracted.py", "src/jev_dllm/evaluation/code_extraction.py",
         "src/jev_dllm/evaluation/humaneval_isolated.py", "src/jev_dllm/evaluation/humaneval_worker.py",
         "configs/evaluation/humaneval-extracted-v2.json")


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_rows(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL record: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def implementation(root):
    return {p: frozen.sha256(root / p) for p in (*FILES, *frozen_selector.pinned_paths())}


def inputs(root, archive):
    record = frozen_selector.check(root)
    audit = collection.audit(root, archive)
    if audit["task"] != "humaneval" or audit["status"] != "complete":
        raise ValueError("Require a complete, verified HumanEval archive")
    questions, pools = selection.load_pools(root / QUESTIONS, archive / "candidates.jsonl")
    if (len(questions) != EXPECTED_QUESTIONS or sum(map(len, pools.values())) != EXPECTED_CANDIDATES
            or any(q.dataset != "humaneval" or q.split != "heldout" for q in questions)):
        raise ValueError("Require the complete 164-task, eight-candidate held-out HumanEval pool")
    prompt_paths = frozen_selector.prompts(BENCHMARK)
    prompts = {k: (root / v).read_text() for k, v in prompt_paths.items()}
    prompt_hashes = {k: frozen.sha256(root / v) for k, v in prompt_paths.items()}
    if any(record["files_sha256"][prompt_paths[k]] != prompt_hashes[k] for k in prompt_paths):
        raise ValueError("HumanEval pointwise prompts differ from the frozen selector record")
    return record, questions, pools, prompt_paths, prompts, audit


def make_requests(questions, pools, prompts, model):
    rows, transformations = [], []
    for q in questions:
        for c in pools[q.question_id]:
            extracted, record = extraction.extraction_record(c)
            transformations.append(record)
            judged = c if extracted.source is None else replace(c, text=extracted.source)
            rows.append({"call_index": len(rows), "question_id": q.question_id,
                         "candidate_id": c.candidate_id,
                         "body": build_pointwise_request(q, judged, prompts["correct"], prompts["consistent"], model)})
    if len(rows) != EXPECTED_CANDIDATES:
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} pointwise requests, got {len(rows)}")
    return rows, transformations


def prepare(root, archive):
    record, questions, pools, prompt_paths, prompts, audit = inputs(root, archive)
    requests, transformations = make_requests(questions, pools, prompts, record["jev"]["model"])
    out = selection.new_run(root / "artifacts/runs", "humaneval-pointwise-plan")
    (out / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests))
    write_json(out / "transformations.json", transformations)
    plan = {"version": VERSION, "mode": "frozen_pointwise_humaneval", "status": "prepared",
            "source_run": archive.relative_to(root).as_posix(),
            "source_candidates_sha256": frozen.sha256(archive / "candidates.jsonl"),
            "source_manifest_sha256": frozen.sha256(archive / "manifest.json"),
            "questions_sha256": frozen.sha256(root / QUESTIONS), "collection_audit": audit,
            "selector_record": frozen_selector.RECORD,
            "selector_record_sha256": frozen.sha256(root / frozen_selector.RECORD),
            "selector": record["selector"], "model": record["jev"]["model"],
            "jev_protocol": record["jev"]["protocol"], "prompt_paths": prompt_paths,
            "prompt_sha256": {k: frozen.sha256(root / v) for k, v in prompt_paths.items()},
            "implementation_sha256": implementation(root),
            "requests_sha256": frozen.sha256(out / "requests.jsonl"),
            "transformations_sha256": frozen.sha256(out / "transformations.json"),
            "request_count": len(requests), "planned_calls": {"pointwise": len(requests), "total": len(requests)},
            "paid_calls_sent": 0, "automatic_retries": 0,
            "note": "Public held-out inputs only; highest correct probability per task, ties by original pool order."}
    write_json(out / "plan.json", plan)
    return out


def check(root, plan_dir):
    p = json.loads((plan_dir / "plan.json").read_text())
    if p.get("version") != VERSION or p.get("mode") != "frozen_pointwise_humaneval":
        raise ValueError("Unknown HumanEval pointwise plan")
    archive = root / p["source_run"]
    record, questions, pools, prompt_paths, prompts, audit = inputs(root, archive)
    requests, transformations = make_requests(questions, pools, prompts, record["jev"]["model"])
    if (p["source_candidates_sha256"] != frozen.sha256(archive / "candidates.jsonl")
            or p["source_manifest_sha256"] != frozen.sha256(archive / "manifest.json")
            or p["questions_sha256"] != frozen.sha256(root / QUESTIONS)
            or p["selector_record_sha256"] != frozen.sha256(root / frozen_selector.RECORD)
            or p["prompt_paths"] != prompt_paths
            or p["prompt_sha256"] != {k: frozen.sha256(root / v) for k, v in prompt_paths.items()}
            or p["implementation_sha256"] != implementation(root)
            or p["requests_sha256"] != frozen.sha256(plan_dir / "requests.jsonl")
            or p["transformations_sha256"] != frozen.sha256(plan_dir / "transformations.json")
            or json.loads((plan_dir / "transformations.json").read_text()) != transformations
            or read_rows(plan_dir / "requests.jsonl") != requests
            or p["request_count"] != EXPECTED_CANDIDATES
            or p["planned_calls"] != {"pointwise": EXPECTED_CANDIDATES, "total": EXPECTED_CANDIDATES}
            or p["paid_calls_sent"] != 0 or p["automatic_retries"] != 0):
        raise ValueError("Plan, requests, archive, or frozen selector inputs changed")
    return p, requests, questions, pools


def live(root, plan_dir, approved_calls, env_file):
    p, requests, questions, pools = check(root, plan_dir)
    if approved_calls != EXPECTED_CANDIDATES:
        raise ValueError(f"Approve exactly {EXPECTED_CANDIDATES} paid requests")
    load_api_key(env_file)
    claims = root / "artifacts/humaneval-pointwise-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = fingerprint({"source": p["source_candidates_sha256"], "requests": p["requests_sha256"]})
    with (claims / f"{claim_id}.json").open("x") as stream:
        json.dump({"plan": str(plan_dir.relative_to(root)), "requests_sha256": p["requests_sha256"]}, stream)
        stream.write("\n")
    out = plan_dir / "live"
    out.mkdir(exist_ok=False)
    manifest = {"status": "running", "plan_sha256": frozen.sha256(plan_dir / "plan.json"),
                "approved_calls": approved_calls, "attempted_calls": 0, "successful_calls": 0,
                "automatic_retries": 0, "api_cost_usd": None}
    write_json(out / "manifest.json", manifest)
    judge = PointwiseJudge()
    try:
        with (out / "candidate-scores.jsonl").open("x") as stream:
            for row in requests:
                manifest["attempted_calls"] += 1
                write_json(out / "manifest.json", manifest)
                try:
                    result = judge.judge(row["body"])
                    parsed = parse_pointwise_response(result["raw_response"], row["body"])
                    if any(result.get(k) != v for k, v in parsed.items()):
                        raise ValueError("Saved score differs from validated pointwise response")
                except BaseException as exc:
                    if hasattr(exc, "raw_response"):
                        write_json(out / "rejected_response.json", {"call_index": row["call_index"],
                            "question_id": row["question_id"], "candidate_id": row["candidate_id"],
                            "raw_response": exc.raw_response, "request_seconds": getattr(exc, "request_seconds", None)})
                    manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                    error_type=type(exc).__name__, failed_call_index=row["call_index"])
                    write_json(out / "manifest.json", manifest)
                    raise
                saved = {k: row[k] for k in ("call_index", "question_id", "candidate_id")}
                saved.update({k: result[k] for k in ("correct_probability", "consistent_probability", "model", "usage", "request_seconds", "raw_response")})
                stream.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                manifest["successful_calls"] += 1
                write_json(out / "manifest.json", manifest)
                if manifest["successful_calls"] % 100 == 0:
                    print(f"HumanEval Jev pointwise: {manifest['successful_calls']}/{EXPECTED_CANDIDATES}", flush=True)
    except BaseException:
        raise
    score_rows = read_rows(out / "candidate-scores.jsonl")
    score_map = {r["candidate_id"]: r["correct_probability"] for r in score_rows}
    if len(score_map) != EXPECTED_CANDIDATES:
        raise ValueError("Pointwise score file does not contain every candidate exactly once")
    selected = []
    for q in questions:
        ids = [c.candidate_id for c in pools[q.question_id]]
        cid = frozen_selector.select(ids, score_map)
        selected.append({"question_id": q.question_id, "candidate_id": cid,
                         "correct_probability": score_map[cid],
                         "rule": "highest_correct_probability_ties_by_pool_order"})
    (out / "selections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    manifest.update(status="complete", successful_calls=EXPECTED_CANDIDATES,
                    selected_questions=len(selected), scores_sha256=frozen.sha256(out / "candidate-scores.jsonl"),
                    selections_sha256=frozen.sha256(out / "selections.jsonl"))
    write_json(out / "manifest.json", manifest)
    return out


def evaluate(root, plan_dir, score_run):
    p, requests, questions, pools = check(root, plan_dir)
    live_dir = plan_dir / "live"
    m = json.loads((live_dir / "manifest.json").read_text())
    scores = read_rows(live_dir / "candidate-scores.jsonl")
    selections = read_rows(live_dir / "selections.jsonl")
    if (m.get("status") != "complete" or m.get("successful_calls") != EXPECTED_CANDIDATES
            or len(scores) != EXPECTED_CANDIDATES or len(selections) != EXPECTED_QUESTIONS
            or m.get("scores_sha256") != frozen.sha256(live_dir / "candidate-scores.jsonl")
            or m.get("selections_sha256") != frozen.sha256(live_dir / "selections.jsonl")):
        raise ValueError("Require a complete, unchanged pointwise run")
    for row, req in zip(scores, requests):
        parsed = parse_pointwise_response(row["raw_response"], req["body"])
        if (row["question_id"] != req["question_id"] or row["candidate_id"] != req["candidate_id"]
                or row["correct_probability"] != parsed["correct_probability"]):
            raise ValueError("Saved pointwise score differs from the planned request")
    labels, baseline = humaneval_jev.score_labels(root, root / p["source_run"], score_run, questions, pools)
    by_candidate = {r["candidate_id"]: r["correct_probability"] for r in scores}
    chosen_labels, first_labels, random_labels = [], [], []
    for q, sel in zip(questions, selections):
        ids = [c.candidate_id for c in pools[q.question_id]]
        expected = frozen_selector.select(ids, by_candidate)
        if sel["question_id"] != q.question_id or sel["candidate_id"] != expected:
            raise ValueError("Saved selected candidate differs from frozen pointwise rule")
        chosen_labels.append(labels[expected])
        first_labels.append(labels[ids[0]])
        random_labels.append(labels[random_choice(pools[q.question_id], trajectory_seed(20261008, q.question_id, 1))])
    correct = sum(chosen_labels)
    baseline.update(jev_evaluated=True, analysis_type="heldout_frozen_pointwise_posthoc", selector="jev_pointwise",
                    jev={"correct": correct, "total": len(questions), "accuracy": correct / len(questions)},
                    jev_vs_first_sample=humaneval_jev.paired_summary(chosen_labels, first_labels, 20261008),
                    jev_vs_seeded_random=humaneval_jev.paired_summary(chosen_labels, random_labels, 20261008),
                    pointwise_calls=len(scores), api_cost_usd=None,
                    request_seconds_sum=sum(r["request_seconds"] for r in scores),
                    usage={k: sum(r["usage"][k] for r in scores) for k in ("input_tokens", "output_tokens")},
                    note="Held-out post hoc scoring using the frozen Week 4 pointwise selector; not an untouched benchmark claim.")
    out = selection.new_run(root / "artifacts/runs", "humaneval-pointwise-evaluation")
    write_json(out / "metrics.json", baseline)
    (out / "summary.md").write_text(
        "# HumanEval held-out pointwise evaluation\n\n"
        f"Questions: {len(questions)}  \nCandidates: {EXPECTED_CANDIDATES}  \nJev pointwise calls: {len(scores)}  \n"
        f"Jev accuracy: {correct}/{len(questions)} ({correct/len(questions):.1%})\n\n"
        f"First candidate: {baseline['first_sample']['accuracy']:.1%}  \n"
        f"Seeded random: {baseline['seeded_random']['accuracy']:.1%}\n\n"
        "This is a post hoc held-out analysis under the pinned extraction/scoring policy.\n")
    write_json(out / "manifest.json", {"status": "complete", "plan": str(plan_dir),
        "score_run": str(score_run), "selector_record": frozen_selector.RECORD,
        "evidence_sha256": {str(f): frozen.sha256(f) for f in (plan_dir / "plan.json", live_dir / "manifest.json",
            live_dir / "candidate-scores.jsonl", live_dir / "selections.jsonl", score_run / "manifest.json",
            score_run / "candidate-results.jsonl")}, "implementation_sha256": implementation(root), "new_api_calls": 0})
    return out, baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "check", "live", "evaluate"))
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--plan-dir", type=Path)
    ap.add_argument("--score-run", type=Path)
    ap.add_argument("--approved-calls", type=int)
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args()
    root = a.root.resolve()
    try:
        if a.action == "prepare":
            if not a.archive: ap.error("prepare requires --archive")
            out = prepare(root, (root / a.archive).resolve())
            p = json.loads((out / "plan.json").read_text())
            result = {"status": "prepared", "plan_directory": str(out), "planned_calls": p["planned_calls"], "paid_calls_sent": 0}
        elif a.action == "check":
            if not a.plan_dir: ap.error("check requires --plan-dir")
            p, req, *_ = check(root, (root / a.plan_dir).resolve())
            result = {"status": "passed", "requests": len(req), "planned_calls": p["planned_calls"], "paid_calls_sent": 0}
        elif a.action == "live":
            if not a.plan_dir or a.approved_calls is None: ap.error("live requires --plan-dir and --approved-calls")
            out = live(root, (root / a.plan_dir).resolve(), a.approved_calls, (root / a.env_file).resolve())
            result = {"status": "complete", "live_directory": str(out)}
        else:
            if not a.plan_dir or not a.score_run: ap.error("evaluate requires --plan-dir and --score-run")
            out, metrics = evaluate(root, (root / a.plan_dir).resolve(), (root / a.score_run).resolve())
            result = {"status": "complete", "evaluation_directory": str(out), "jev": metrics["jev"]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"HumanEval pointwise stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
