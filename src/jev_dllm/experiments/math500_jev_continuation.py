"""One explicit retry amendment for a failed MATH500 Jev batch; no silent retries."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from . import math500_jev as original
from .selection import new_run, write_json
from ..judges.typesafe import TypeSafeJudge, parse_response
from ..reproducibility import fingerprint

VERSION = "math500_jev_one_retry_v1"
FILES = ("src/jev_dllm/experiments/math500_jev_continuation.py",
         "scripts/continue_math500_jev.py", "Makefile.math500-jev-recovery")
sha256 = original.frozen.sha256


def implementation(root):
    for name in FILES:
        if any((root / part).is_symlink() for part in (Path(name), *Path(name).parents)):
            raise ValueError("Continuation implementation must use ordinary files")
    return {name: sha256(root / name) for name in FILES}


def inventory(directory):
    paths = sorted(directory.rglob("*"))
    if directory.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError("Continuation evidence must use ordinary files")
    return {str(p.relative_to(directory)): sha256(p) for p in paths if p.is_file()}


def validate_selections(rows, requests):
    if len(rows) > len(requests):
        raise ValueError("Too many saved selections")
    for row, request in zip(rows, requests):
        parsed = parse_response(row["raw_response"], request["body"])
        if row["question_id"] != request["question_id"] or any(row.get(k) != v for k,v in parsed.items()):
            raise ValueError("Saved selection is not the exact validated request prefix")
        seconds = row["request_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid saved request latency")


def failed_source(root, source):
    p, requests, questions, pools = original.inspect(root, source)
    live = source / "live"
    m = json.loads((live / "manifest.json").read_text())
    accepted = original.read_rows(live / "selections.jsonl")
    n = len(accepted)
    if (m["status"] != "failed" or m["error_type"] != "ValueError"
            or m["plan_sha256"] != sha256(source / "plan.json")
            or not 0 <= n < len(requests) or m["successful_calls"] != n
            or m["attempted_calls"] != n + 1 or m["approved_calls"] != len(requests)
            or m["automatic_retries"] != 0):
        raise ValueError("Require an original failed batch with one rejected response after a valid prefix")
    validate_selections(accepted, requests)
    failure = json.loads((live / "failed_response.json").read_text())
    if failure["question_id"] != requests[n]["question_id"] or m["failed_question_id"] != failure["question_id"]:
        raise ValueError("Recorded failure does not follow the saved prefix")
    raw = json.loads(failure["raw_response_json"])
    try:
        parse_response(raw, requests[n]["body"])
    except ValueError as exc:
        if str(exc) != "Jev choice is inconsistent with its probabilities":
            raise ValueError("Only a Choice/probability mismatch supports this amendment") from exc
    else:
        raise ValueError("Recorded mismatch no longer reproduces")
    # Check fields validated after the mismatch. This copy is never accepted as a selection.
    check = deepcopy(raw)
    answer = check["answers"]["best_answer"]
    answer["choice"] = max(answer["probabilities"], key=answer["probabilities"].get)
    parse_response(check, requests[n]["body"])
    seconds = failure["request_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid rejected request latency")
    return p, requests, questions, pools, accepted, failure


def prepare(root, source):
    p, requests, _, _, accepted, _ = failed_source(root, source)
    out = new_run(root / "artifacts/runs", "math500-jev-continuation-plan")
    (out / "planned_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests[len(accepted):]))
    write_json(out / "plan.json", {"version": VERSION, "source_plan": str(source.resolve()),
        "source_files_sha256": inventory(source), "implementation_sha256": implementation(root),
        "planned_requests_sha256": sha256(out / "planned_requests.jsonl"),
        "reused_selections": len(accepted), "prior_attempted_calls": len(accepted) + 1,
        "additional_paid_calls": len(requests) - len(accepted), "unattempted_questions": len(requests)-len(accepted)-1,
        "failed_question_id": requests[len(accepted)]["question_id"],
        "total_attempts_if_successful": len(requests)+1, "explicit_retries": 1, "automatic_retries": 0,
        "note": "Proposed amendment only. Live execution requires approval for this additional request count."})
    return out


def inspect(root, plan):
    amendment = json.loads((plan / "plan.json").read_text())
    source = Path(amendment["source_plan"])
    if (amendment["version"] != VERSION or amendment["implementation_sha256"] != implementation(root)
            or inventory(source) != amendment["source_files_sha256"]):
        raise ValueError("Continuation implementation or original evidence changed")
    details = failed_source(root, source)
    _, requests, _, _, accepted, _ = details
    n, total = len(accepted), len(requests)
    expected = {"reused_selections": n, "prior_attempted_calls": n+1,
                "additional_paid_calls": total-n, "unattempted_questions": total-n-1,
                "failed_question_id": requests[n]["question_id"],
                "total_attempts_if_successful": total+1, "explicit_retries": 1, "automatic_retries": 0}
    if (any(amendment.get(k) != v for k,v in expected.items())
            or sha256(plan / "planned_requests.jsonl") != amendment["planned_requests_sha256"]
            or original.read_rows(plan / "planned_requests.jsonl") != requests[n:]):
        raise ValueError("Continuation must contain the unchanged failed request and unattempted suffix")
    return amendment, source, details


def execute(root, plan, approved_calls, client=None):
    amendment, source, (_, requests, _, _, accepted, _) = inspect(root, plan)
    if type(approved_calls) is not int or approved_calls != amendment["additional_paid_calls"]:
        raise ValueError("Explicitly approve exactly this planned additional request count")
    claims = root / "artifacts/math500-jev-continuation-claims"
    claims.mkdir(parents=True, exist_ok=True)
    # All plans for this original failed batch share one claim, including freshly prepared plans.
    with (claims / (fingerprint(amendment["source_files_sha256"]) + ".json")).open("x") as stream:
        stream.write(json.dumps({"plan": str(plan), "plan_sha256": sha256(plan / "plan.json")}) + "\n")
    out = plan / "live"
    out.mkdir(exist_ok=False)
    m = {"status": "running", "amendment_plan_sha256": sha256(plan / "plan.json"),
         "reused_selections": len(accepted), "attempted_calls": 0, "newly_successful_calls": 0,
         "successful_calls": len(accepted), "total_attempted_calls": amendment["prior_attempted_calls"],
         "approved_additional_calls": approved_calls, "automatic_retries": 0, "api_cost_usd": None}
    write_json(out / "manifest.json", m)
    print(f"MATH500 continuation directory: {out}", flush=True)
    try:
        shutil.copyfile(source / "live/selections.jsonl", out / "selections.jsonl")
        shutil.copyfile(source / "live/failed_response.json", out / "prior_failed_response.json")
        shutil.copyfile(source / "live/manifest.json", out / "prior_manifest.json")
        client = client or TypeSafeJudge()
        with (out / "selections.jsonl").open("a") as stream:
            for request in requests[len(accepted):]:
                m["attempted_calls"] += 1
                m["total_attempted_calls"] += 1
                write_json(out / "manifest.json", m)
                result = {"question_id": request["question_id"], **client.select(request["body"])}
                validate_selections([result], [request])
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                m["newly_successful_calls"] += 1
                m["successful_calls"] += 1
                write_json(out / "manifest.json", m)
                print(f"MATH500 Jev: {m['successful_calls']}/{len(requests)}; additional requests: {m['attempted_calls']}/{approved_calls}", flush=True)
        inspect(root, plan)
        if not (out / "selections.jsonl").read_bytes().startswith((source / "live/selections.jsonl").read_bytes()):
            raise ValueError("Original accepted selection bytes changed")
        m.update(status="complete", selections_sha256=sha256(out / "selections.jsonl"))
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        if "request" in locals():
            m["failed_question_id"] = request["question_id"]
        if hasattr(exc, "raw_response"):
            write_json(out / "failed_response.json", {"question_id": request["question_id"],
                "raw_response_json": json.dumps(exc.raw_response, ensure_ascii=False), "request_seconds": exc.request_seconds})
        raise
    finally:
        write_json(out / "manifest.json", m)
    return out


def audit_live(root, plan):
    """Check the amended selection and every billed attempt; correctness is separate."""
    amendment, source, (_, requests, questions, pools, accepted, failure) = inspect(root, plan)
    live = plan / "live"
    m = json.loads((live / "manifest.json").read_text())
    rows = original.read_rows(live / "selections.jsonl")
    expected = {"status": "complete", "amendment_plan_sha256": sha256(plan / "plan.json"),
        "reused_selections": len(accepted), "attempted_calls": amendment["additional_paid_calls"],
        "newly_successful_calls": amendment["additional_paid_calls"], "successful_calls": len(requests),
        "total_attempted_calls": len(requests)+1, "approved_additional_calls": amendment["additional_paid_calls"],
        "automatic_retries": 0, "selections_sha256": sha256(live / "selections.jsonl")}
    if (any(m.get(k) != v for k,v in expected.items()) or len(rows) != len(requests)
            or not (live / "selections.jsonl").read_bytes().startswith((source / "live/selections.jsonl").read_bytes())):
        raise ValueError("Require a complete continuation with the unchanged original selection prefix")
    for saved, original_name in (("prior_failed_response.json", "failed_response.json"),
                                 ("prior_manifest.json", "manifest.json")):
        if (live / saved).read_bytes() != (source / "live" / original_name).read_bytes():
            raise ValueError("Copied prior failure evidence changed")
    validate_selections(rows, requests)
    selected = []
    for row in rows:
        c = next(c for c in pools[row["question_id"]] if c.candidate_id == row["candidate_id"])
        selected.append({"question_id": row["question_id"], "candidate_id": c.candidate_id, "text": c.text})
    failed_usage = json.loads(failure["raw_response_json"])["usage"]
    result = {"status": "complete", "selection_version": VERSION,
        "analysis_type": "raw_answer_selection_with_explicit_retry",
        "question_count": len(questions), "candidate_count": sum(len(v) for v in pools.values()),
        "accepted_selections": len(rows), "total_attempted_calls": m["total_attempted_calls"],
        "rejected_responses": 1, "explicit_retries": 1, "automatic_retries": 0,
        "api_cost_usd": None, "request_seconds_sum_including_rejected": sum(r["request_seconds"] for r in rows)+failure["request_seconds"],
        "usage_including_rejected": {k: sum(r["usage"][k] for r in rows)+failed_usage[k] for k in ("input_tokens", "output_tokens")},
        "accuracy": None, "correctness_scored": False,
        "note": "One explicit retry amendment; original accepted responses and rejected evidence retained. Mathematical correctness scoring remains separate."}
    out = new_run(root / "artifacts/runs", "math500-jev-amended-selection-audit")
    write_json(out / "selection-summary.json", result)
    (out / "selected-answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    write_json(out / "manifest.json", {"status": "complete", "amendment": amendment,
        "implementation_sha256": implementation(root), "new_api_calls": 0, "correctness_scored": False,
        "evidence_sha256": {str(f): sha256(f) for f in (plan / "plan.json", live / "manifest.json",
            live / "selections.jsonl", live / "prior_failed_response.json", live / "prior_manifest.json")},
        "outputs_sha256": {name: sha256(out / name) for name in ("selection-summary.json", "selected-answers.jsonl")}})
    print(json.dumps(result, indent=2))
    return out
