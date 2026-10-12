"""Explicit continuation of failed Jev continuations, retaining every attempt."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from . import math500_jev_continuation as previous
from .selection import new_run, write_json
from ..judges.typesafe import TypeSafeJudge, parse_response
from ..reproducibility import fingerprint

original = previous.original
sha256 = previous.sha256
VERSION = "math500_jev_explicit_resume_v1"
FILES = (*previous.FILES, "src/jev_dllm/experiments/math500_jev_resume.py",
         "scripts/resume_math500_jev.py", "Makefile")


def implementation(root):
    for name in FILES:
        if any((root / part).is_symlink() for part in (Path(name), *Path(name).parents)):
            raise ValueError("Resume implementation must use ordinary files")
    return {name: sha256(root / name) for name in FILES}


def validate_failure(failure, request):
    if failure["question_id"] != request["question_id"]:
        raise ValueError("Failure must follow the exact accepted prefix")
    raw = json.loads(failure["raw_response_json"])
    try:
        parse_response(raw, request["body"])
    except ValueError as exc:
        if str(exc) != "Jev choice is inconsistent with its probabilities":
            raise ValueError("Only a Choice/probability mismatch supports explicit resume") from exc
    else:
        raise ValueError("Recorded mismatch no longer reproduces")
    # Check later schema fields only; this copy is never saved or used as a choice.
    check = deepcopy(raw)
    answer = check["answers"]["best_answer"]
    answer["choice"] = max(answer["probabilities"], key=answer["probabilities"].get)
    parse_response(check, request["body"])
    seconds = failure["request_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid rejected request latency")


def _check_segment(plan, amendment, source, prior, *, failed):
    """Validate one execution segment against its independently checked ancestors."""
    live = plan / "live"
    m = json.loads((live / "manifest.json").read_text())
    rows = original.read_rows(live / "selections.jsonl")
    n, reused, total = len(rows), len(prior["accepted"]), len(prior["requests"])
    if not reused <= n <= total or (failed and n == total) or (not failed and n != total):
        raise ValueError("Invalid saved selection count for this segment")
    attempts = n - reused + int(failed)
    expected = {"status": "failed" if failed else "complete",
        "amendment_plan_sha256": sha256(plan / "plan.json"), "reused_selections": reused,
        "attempted_calls": attempts, "newly_successful_calls": n - reused,
        "successful_calls": n, "total_attempted_calls": prior["total_attempts"] + attempts,
        "approved_additional_calls": total - reused, "automatic_retries": 0}
    if failed:
        expected.update(error_type="ValueError", failed_question_id=prior["requests"][n]["question_id"])
    else:
        expected["selections_sha256"] = sha256(live / "selections.jsonl")
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError("Segment status, approval or attempt accounting changed")
    previous.validate_selections(rows, prior["requests"])
    if not (live / "selections.jsonl").read_bytes().startswith((source / "live/selections.jsonl").read_bytes()):
        raise ValueError("Accepted source bytes changed")
    for saved, name in (("prior_manifest.json", "manifest.json"),
                        ("prior_failed_response.json", "failed_response.json")):
        if (live / saved).read_bytes() != (source / "live" / name).read_bytes():
            raise ValueError("Copied prior evidence changed")
    if amendment["version"] == VERSION:
        if json.loads((live / "prior_failures.json").read_text()) != prior["failures"]:
            raise ValueError("Rejected-attempt history changed")
    failures = list(prior["failures"])
    if failed:
        failure = json.loads((live / "failed_response.json").read_text())
        validate_failure(failure, prior["requests"][n])
        failures.append({"source_plan": str(plan.resolve()), "response": failure})
    if m["total_attempted_calls"] != n + len(failures):
        raise ValueError("Every attempted call must have accepted or rejected evidence")
    return {**prior, "accepted": rows, "failures": failures,
            "total_attempts": m["total_attempted_calls"]}


def failed_source(root, source, seen=frozenset()):
    if source.resolve() in seen:
        raise ValueError("Cyclic continuation ancestry")
    previous.inventory(source)  # Reject symlinks before reading evidence.
    version = json.loads((source / "plan.json").read_text())["version"]
    if version == previous.VERSION:
        amendment, parent, (_, requests, questions, pools, accepted, failure) = previous.inspect(root, source)
        prior = {"requests": requests, "questions": questions, "pools": pools,
                 "accepted": accepted, "total_attempts": len(accepted) + 1,
                 "failures": [{"source_plan": str(parent.resolve()), "response": failure}]}
    elif version == VERSION:
        amendment, parent, prior = inspect(root, source, seen)
    else:
        raise ValueError("Source must be a failed MATH500 Jev continuation")
    return _check_segment(source, amendment, parent, prior, failed=True)


def _counts(details):
    n, total, rejected = len(details["accepted"]), len(details["requests"]), len(details["failures"])
    return {"reused_selections": n, "prior_attempted_calls": details["total_attempts"],
        "prior_rejected_responses": rejected, "additional_paid_calls": total - n,
        "unattempted_questions": total - n - 1, "failed_question_id": details["requests"][n]["question_id"],
        "total_attempts_if_successful": total + rejected,
        "explicit_retries_if_successful": rejected, "automatic_retries": 0}


def prepare(root, source):
    details = failed_source(root, source)
    out = new_run(root / "artifacts/runs", "math500-jev-resume-plan")
    remaining = details["requests"][len(details["accepted"]):]
    (out / "planned_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in remaining))
    write_json(out / "plan.json", {"version": VERSION, "source_plan": str(source.resolve()),
        "source_files_sha256": previous.inventory(source), "implementation_sha256": implementation(root),
        "planned_requests_sha256": sha256(out / "planned_requests.jsonl"), **_counts(details),
        "note": "Offline proposal only; live execution requires the exact additional paid-call approval."})
    return out


def inspect(root, plan, seen=frozenset()):
    if plan.resolve() in seen:
        raise ValueError("Cyclic continuation ancestry")
    previous.inventory(plan)
    amendment = json.loads((plan / "plan.json").read_text())
    source = Path(amendment["source_plan"])
    if (amendment["version"] != VERSION or amendment["implementation_sha256"] != implementation(root)
            or previous.inventory(source) != amendment["source_files_sha256"]):
        raise ValueError("Resume implementation or source evidence changed")
    details = failed_source(root, source, seen | {plan.resolve()})
    if (any(amendment.get(k) != v for k, v in _counts(details).items())
            or sha256(plan / "planned_requests.jsonl") != amendment["planned_requests_sha256"]
            or original.read_rows(plan / "planned_requests.jsonl") != details["requests"][len(details["accepted"]):]):
        raise ValueError("Resume requires the unchanged failed request and unattempted suffix")
    return amendment, source, details


def execute(root, plan, approved_calls, client=None):
    amendment, source, prior = inspect(root, plan)
    if type(approved_calls) is not int or approved_calls != amendment["additional_paid_calls"]:
        raise ValueError("Explicitly approve exactly this planned additional request count")
    claims = root / "artifacts/math500-jev-resume-claims"
    claims.mkdir(parents=True, exist_ok=True)
    # One launch per failed source, even if another plan is prepared for it.
    with (claims / (fingerprint(str(source.resolve())) + ".json")).open("x") as stream:
        stream.write(json.dumps({"plan": str(plan), "plan_sha256": sha256(plan / "plan.json")}) + "\n")
    out = plan / "live"
    out.mkdir(exist_ok=False)
    m = {"status": "running", "amendment_plan_sha256": sha256(plan / "plan.json"),
        "reused_selections": len(prior["accepted"]), "attempted_calls": 0, "newly_successful_calls": 0,
        "successful_calls": len(prior["accepted"]), "total_attempted_calls": prior["total_attempts"],
        "approved_additional_calls": approved_calls, "automatic_retries": 0, "api_cost_usd": None}
    write_json(out / "manifest.json", m)
    print(f"MATH500 resume directory: {out}", flush=True)
    try:
        shutil.copyfile(source / "live/selections.jsonl", out / "selections.jsonl")
        shutil.copyfile(source / "live/manifest.json", out / "prior_manifest.json")
        shutil.copyfile(source / "live/failed_response.json", out / "prior_failed_response.json")
        write_json(out / "prior_failures.json", prior["failures"])
        client = client or TypeSafeJudge()
        with (out / "selections.jsonl").open("a") as stream:
            for request in prior["requests"][len(prior["accepted"]):]:
                m["attempted_calls"] += 1
                m["total_attempted_calls"] += 1
                write_json(out / "manifest.json", m)
                result = {"question_id": request["question_id"], **client.select(request["body"])}
                previous.validate_selections([result], [request])
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                m["newly_successful_calls"] += 1
                m["successful_calls"] += 1
                write_json(out / "manifest.json", m)
                print(f"MATH500 Jev: {m['successful_calls']}/{len(prior['requests'])}; additional requests: {m['attempted_calls']}/{approved_calls}", flush=True)
        inspect(root, plan)
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
    amendment, source, prior = inspect(root, plan)
    details = _check_segment(plan, amendment, source, prior, failed=False)
    rows, failures, pools = details["accepted"], details["failures"], details["pools"]
    all_usage = [r["usage"] for r in rows] + [json.loads(f["response"]["raw_response_json"])["usage"] for f in failures]
    result = {"status": "complete", "selection_version": VERSION,
        "analysis_type": "raw_answer_selection_with_explicit_retries",
        "question_count": len(details["questions"]), "candidate_count": sum(len(v) for v in pools.values()),
        "accepted_selections": len(rows), "total_attempted_calls": details["total_attempts"],
        "rejected_responses": len(failures), "explicit_retries": len(failures), "automatic_retries": 0,
        "usage_including_rejected": {k: sum(u[k] for u in all_usage) for k in ("input_tokens", "output_tokens")},
        "request_seconds_sum_including_rejected": sum(r["request_seconds"] for r in rows) + sum(f["response"]["request_seconds"] for f in failures),
        "api_cost_usd": None, "accuracy": None, "correctness_scored": False,
        "note": "Each failed segment required a separate explicit continuation. Mathematical correctness scoring remains separate."}
    selected = []
    for row in rows:
        c = next(c for c in pools[row["question_id"]] if c.candidate_id == row["candidate_id"])
        selected.append({"question_id": row["question_id"], "candidate_id": c.candidate_id, "text": c.text})
    out = new_run(root / "artifacts/runs", "math500-jev-resumed-selection-audit")
    write_json(out / "selection-summary.json", result)
    write_json(out / "rejected-attempts.json", failures)
    (out / "selected-answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    write_json(out / "manifest.json", {"status": "complete", "amendment": amendment,
        "implementation_sha256": implementation(root), "new_api_calls": 0, "correctness_scored": False,
        "evidence_sha256": previous.inventory(plan),
        "outputs_sha256": {name: sha256(out / name) for name in ("selection-summary.json", "selected-answers.jsonl", "rejected-attempts.json")}})
    print(json.dumps(result, indent=2))
    return out
