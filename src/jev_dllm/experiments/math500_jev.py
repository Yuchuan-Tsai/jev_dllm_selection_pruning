"""MATH500 public-only Jev selection; mathematical correctness grading is separate."""
import json
import math
from pathlib import Path
import platform
import random

from . import collection_recovery, frozen
from .selection import load_pools, new_run, write_json
from ..judges.typesafe import TypeSafeJudge, parse_response
from ..reproducibility import fingerprint, trajectory_seed
from ..operations.remote import snapshot_digest, valid_run_id

VERSION = "math500_raw_jev_v1"
MODEL = "jev-1.13.0"
SEED = 20261008
PROMPT = "prompts/jev_final.txt"
FILES = ("src/jev_dllm/experiments/math500_jev.py", "scripts/judge_math500.py", "Makefile.math500", PROMPT,
         "src/jev_dllm/judges/typesafe.py", "src/jev_dllm/judges/credentials.py",
         "src/jev_dllm/experiments/selection.py", "src/jev_dllm/reproducibility.py",
         "src/jev_dllm/schema.py", "src/jev_dllm/operations/remote.py") + collection_recovery.EXTENSION_FILES


def read_rows(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Incomplete final record; retain the evidence")
    return [json.loads(line) for line in raw.splitlines()]


def implementation(root):
    for name in FILES:
        if any((root / part).is_symlink() for part in (Path(name), *Path(name).parents)):
            raise ValueError("Selection implementation must not contain symlinks")
    return {p: frozen.sha256(root / p) for p in FILES}


def public_inputs(root, archive):
    info = collection_recovery.inspect_run(root, archive)
    if info["manifest"]["task"] != "math500" or info["manifest"]["status"] != "complete":
        raise ValueError("Require a complete, verified MATH500 archive")
    questions, pools = load_pools(root / "data/processed/math500-heldout.json", archive / "candidates.jsonl")
    if len(questions) != 500 or any(len(p) != 8 for p in pools.values()):
        raise ValueError("Require all 500 questions and eight candidates per question")
    return questions, pools


def request_rows(questions, pools, prompt):
    """Preserve raw text and all IDs, including empty/incomplete/truncated answers."""
    if not prompt.strip():
        raise ValueError("Selection instructions cannot be empty")
    requests = []
    for q in questions:
        candidates = list(pools[q.question_id])
        if (not 2 <= len(candidates) <= 255
                or len({c.candidate_id for c in candidates}) != len(candidates)
                or any(c.question_id != q.question_id or not isinstance(c.text, str) for c in candidates)):
            raise ValueError("Invalid public candidate identities/text")
        random.Random(trajectory_seed(SEED, q.question_id, 0)).shuffle(candidates)
        # The older final_request helper rejects truncated outputs. This independent
        # wire adapter retains those outputs without changing their recorded flags.
        body = {"model": MODEL,
                "state": {"stage": "final", "question": q.prompt,
                          "candidates": [{"candidate_id": c.candidate_id, "text": c.text} for c in candidates]},
                "questions": {"best_answer": {"type": "choice", "instructions": prompt,
                    "criteria": {c.candidate_id: f"Select candidate {c.candidate_id} from state.candidates."
                                 for c in candidates}}}}
        requests.append({"question_id": q.question_id, "body": body})
    return requests


def prepare(root, archive, destination=None):
    questions, pools = public_inputs(root, archive)
    requests = request_rows(questions, pools, (root / PROMPT).read_text())
    # Stable explicit destinations simplify manual commands without replaying paid work.
    if destination is not None:
        base = root / "artifacts/runs"
        valid_run_id(destination.name)
        if destination.is_symlink() or destination.resolve().parent != base.resolve():
            raise ValueError("Plan directory must be directly under artifacts/runs")
        if destination.exists():
            existing, _, _, _ = inspect(root, destination)
            if Path(existing["source_run"]).resolve() != archive.resolve():
                raise ValueError("Existing plan belongs to a different archive")
            return destination
        destination.mkdir(parents=True, exist_ok=False)
        out = destination
    else:
        out = new_run(root / "artifacts/runs", "math500-jev-plan")
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    plan = {"version": VERSION, "model": MODEL, "seed": SEED,
            "source_run": str(archive.resolve()), "source_candidates_sha256": frozen.sha256(archive / "candidates.jsonl"),
            "source_snapshot_digest": snapshot_digest(collection_recovery.receipt_for(archive)),
            "question_count": len(questions), "candidate_count": sum(len(p) for p in pools.values()),
            "requests_sha256": frozen.sha256(out / "requests.jsonl"),
            "implementation_sha256": implementation(root), "host_python": platform.python_version(),
            "analysis_type": "raw_completed_answer_selection", "automatic_retries": 0,
            "candidate_policy": "All original IDs and raw text retained unchanged; no filtering, extraction or repair.",
            "correctness_scored": False, "paid_calls_sent": 0}
    write_json(out / "plan.json", plan)
    return out


def inspect(root, plan_dir):
    p = json.loads((plan_dir / "plan.json").read_text())
    if (p["version"] != VERSION or p["model"] != MODEL or p["seed"] != SEED
            or p["automatic_retries"] != 0 or p["implementation_sha256"] != implementation(root)
            or p["host_python"] != platform.python_version()
            or p["correctness_scored"] is not False
            or p["analysis_type"] != "raw_completed_answer_selection"):
        raise ValueError("Prepared policy/runtime changed; do not execute this plan")
    archive = Path(p["source_run"])
    questions, pools = public_inputs(root, archive)
    requests = request_rows(questions, pools, (root / PROMPT).read_text())
    if (p["source_candidates_sha256"] != frozen.sha256(archive / "candidates.jsonl")
            or p["source_snapshot_digest"] != snapshot_digest(collection_recovery.receipt_for(archive))
            or p["requests_sha256"] != frozen.sha256(plan_dir / "requests.jsonl")
            or read_rows(plan_dir / "requests.jsonl") != requests
            or p["question_count"] != len(questions) or p["candidate_count"] != sum(len(v) for v in pools.values())):
        raise ValueError("Prepared public inputs changed")
    return p, requests, questions, pools


def validate_result(result, request):
    parsed = parse_response(result["raw_response"], request)
    if any(result.get(k) != v for k, v in parsed.items()):
        raise ValueError("Saved selection differs from validated response")
    seconds = result["request_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid request latency")
    return parsed


def execute(root, plan_dir, approved_calls, client=None):
    p, requests, _, _ = inspect(root, plan_dir)
    if type(approved_calls) is not int or approved_calls != len(requests):
        raise ValueError("Approve exactly the number of paid requests in this plan")
    # Same public batch cannot be replayed by preparing a second directory.
    claims = root / "artifacts/math500-jev-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = fingerprint({"source": p["source_candidates_sha256"], "requests": p["requests_sha256"]})
    out = plan_dir / "live"
    with (claims / (claim_id + ".json")).open("x") as claim:
        claim.write(json.dumps({"plan": str(plan_dir), "output": str(out)}) + "\n")
    out.mkdir(exist_ok=False)
    m = {"status": "running", "plan_sha256": frozen.sha256(plan_dir / "plan.json"),
         "attempted_calls": 0, "successful_calls": 0, "approved_calls": approved_calls,
         "api_cost_usd": None, "automatic_retries": 0}
    write_json(out / "manifest.json", m)
    print(f"Jev output directory: {out}", flush=True)
    client = client or TypeSafeJudge()
    try:
        with (out / "selections.jsonl").open("x") as stream:
            for row in requests:
                m["attempted_calls"] += 1
                write_json(out / "manifest.json", m)
                result = client.select(row["body"])
                validate_result(result, row["body"])
                stream.write(json.dumps({"question_id": row["question_id"], **result}, allow_nan=False) + "\n")
                stream.flush()
                m["successful_calls"] += 1
                write_json(out / "manifest.json", m)
                print(f"MATH500 Jev: {m['successful_calls']}/{len(requests)}", flush=True)
        inspect(root, plan_dir)
        m.update(status="complete", selections_sha256=frozen.sha256(out / "selections.jsonl"))
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        if "row" in locals():
            m["failed_question_id"] = row["question_id"]
        if hasattr(exc, "raw_response"):
            write_json(out / "failed_response.json", {"question_id": row["question_id"],
                "raw_response_json": json.dumps(exc.raw_response, ensure_ascii=False), "request_seconds": exc.request_seconds})
        raise
    finally:
        write_json(out / "manifest.json", m)
    return out


def audit_live(root, plan_dir):
    """Validate saved selections and cost accounting; this does not grade math."""
    p, requests, questions, pools = inspect(root, plan_dir)
    live = plan_dir / "live"
    m = json.loads((live / "manifest.json").read_text())
    selections = read_rows(live / "selections.jsonl")
    if (m["status"] != "complete" or m["plan_sha256"] != frozen.sha256(plan_dir / "plan.json")
            or m["attempted_calls"] != len(questions) or m["successful_calls"] != len(questions)
            or m["approved_calls"] != len(questions) or len(selections) != len(questions)
            or m["automatic_retries"] != 0
            or m["selections_sha256"] != frozen.sha256(live / "selections.jsonl")):
        raise ValueError("Require a complete, unchanged Jev batch; retain partial evidence after any failure")
    selected = []
    for result, request in zip(selections, requests):
        validate_result(result, request["body"])
        if result["question_id"] != request["question_id"]:
            raise ValueError("Selection question order differs")
        candidate = next(c for c in pools[result["question_id"]] if c.candidate_id == result["candidate_id"])
        selected.append({"question_id": result["question_id"], "candidate_id": candidate.candidate_id,
                         "text": candidate.text})
    result = {"status": "complete", "selection_version": VERSION,
              "question_count": len(questions), "candidate_count": p["candidate_count"],
              "accepted_selections": len(selections), "attempted_calls": m["attempted_calls"],
              "automatic_retries": 0, "api_cost_usd": None,
              "request_seconds_sum": sum(r["request_seconds"] for r in selections),
              "usage": {key: sum(r["usage"][key] for r in selections) for key in ("input_tokens", "output_tokens")},
              "correctness_scored": False, "accuracy": None,
              "note": "Selection integrity and API usage only. Validate a separate mathematical-equivalence scorer before reporting accuracy."}
    out = new_run(root / "artifacts/runs", "math500-jev-selection-audit")
    write_json(out / "selection-summary.json", result)
    (out / "selected-answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    write_json(out / "manifest.json", {"status": "complete", "plan": str(plan_dir),
        "evidence_sha256": {str(f): frozen.sha256(f) for f in
            (plan_dir / "plan.json", live / "manifest.json", live / "selections.jsonl")},
        "implementation_sha256": implementation(root), "new_api_calls": 0, "correctness_scored": False,
        "outputs_sha256": {name: frozen.sha256(out / name) for name in ("selection-summary.json", "selected-answers.jsonl")}})
    print(json.dumps(result, indent=2))
    return out
