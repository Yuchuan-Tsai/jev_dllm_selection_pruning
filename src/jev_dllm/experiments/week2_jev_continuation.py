"""Finish a Week 2 Jev live run that stopped on an error, without resending accepted calls.

The source run's accepted prefix is re-validated against the unchanged plan and copied
byte for byte; only the stopped call and the calls after it are sent, once each, after
an exact approval of that count. The stopped call's billing is unknown and is recorded.
Kept separate from week2_jev.py, whose hash is pinned into every prepared plan.
"""
import json
from pathlib import Path
import time

from . import frozen, week2_jev as jev
from .selection import write_json
from ..judges.credentials import load_api_key
from ..judges.listwise import ListwiseJudge
from ..judges.pointwise import PointwiseJudge
from ..reproducibility import fingerprint

OUTPUT = "live-continued"
STOPPED = {"failed", "interrupted"}


def inspect(root: Path, plan_dir: Path) -> dict:
    """Offline: check the plan and the stopped run; return what a continuation would send."""
    p, rows = jev.check(root, plan_dir)
    source = plan_dir / "live"
    m = json.loads((source / "manifest.json").read_text())
    if m.get("status") not in STOPPED:
        raise ValueError("Continue only a failed or interrupted live run")
    if (plan_dir / OUTPUT).exists():
        raise ValueError("This live run already has a continuation")
    raw = (source / "results.jsonl").read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Source results end in a partial record; inspect before continuing")
    records = [json.loads(line) for line in raw.splitlines()]
    for i, (record, row) in enumerate(zip(records, rows)):
        jev.validate_result(record, row)
        key = "subset_size" if row["kind"] == "listwise" else "candidate_id"
        if (record["call_index"], record["kind"], record["question_id"]) != (i, row["kind"], row["question_id"]) \
                or record[key] != row[key]:
            raise ValueError(f"Accepted record {i} does not match its planned request")
    accepted = len(records)
    if m.get("successful_calls") != accepted or m.get("failed_call_index") != accepted:
        raise ValueError("Source manifest disagrees with its saved results")
    return {"plan": p, "rows": rows, "records": records, "source_manifest": m,
            "accepted_calls": accepted, "additional_calls": len(rows) - accepted,
            "retried_call_index": accepted, "source_error": m.get("error_type"),
            "source_results_sha256": frozen.sha256(source / "results.jsonl")}


def live(root: Path, plan_dir: Path, approved_calls: int, env_file: Path | None = None,
         listwise_judge=None, pointwise_judge=None) -> Path:
    """Paid: sends the remaining calls only when approved_calls equals their exact count."""
    info = inspect(root, plan_dir)
    if type(approved_calls) is not int or approved_calls != info["additional_calls"]:
        raise ValueError(f"Approve exactly the {info['additional_calls']} additional paid requests")
    if listwise_judge is None or pointwise_judge is None:
        load_api_key(env_file)
    listwise_judge = listwise_judge or ListwiseJudge()
    pointwise_judge = pointwise_judge or PointwiseJudge()
    claims = root / jev.CLAIMS
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = fingerprint({"version": jev.VERSION, "requests": info["plan"]["requests_sha256"],
                            "continuation_of": info["source_results_sha256"]})
    out = plan_dir / OUTPUT
    try:
        with (claims / (claim_id + ".json")).open("x") as claim:
            claim.write(json.dumps({"plan": str(plan_dir), "output": str(out)}) + "\n")
    except FileExistsError:
        raise ValueError("This continuation was already launched") from None
    out.mkdir()
    rows, records = info["rows"], list(info["records"])
    m = {"status": "running", "task": info["plan"]["task"], "plan_sha256": frozen.sha256(plan_dir / "plan.json"),
         "continued_from": {"output": "live", "status": info["source_manifest"]["status"],
                            "error_type": info["source_error"], "results_sha256": info["source_results_sha256"],
                            "copied_calls": info["accepted_calls"], "retried_call_index": info["retried_call_index"],
                            "billing_unknown_calls": [info["retried_call_index"]]},
         "approved_additional_calls": approved_calls, "attempted_additional_calls": 0,
         "successful_calls": info["accepted_calls"], "api_cost_usd": None, "automatic_retries": 0}
    write_json(out / "manifest.json", m)
    print(f"Jev continuation directory: {out}", flush=True)
    try:
        with (out / "results.jsonl").open("x") as stream:
            stream.write((plan_dir / "live/results.jsonl").read_text())
            stream.flush()
            for row in rows[info["accepted_calls"]:]:
                m["attempted_additional_calls"] += 1
                write_json(out / "manifest.json", m)
                started = time.perf_counter()
                judge = listwise_judge.select if row["kind"] == "listwise" else pointwise_judge.judge
                result = judge(row["body"])
                wall = time.perf_counter() - started
                jev.validate_result(result, row)
                key = {"subset_size": row["subset_size"]} if row["kind"] == "listwise" else {"candidate_id": row["candidate_id"]}
                record = {"call_index": row["call_index"], "kind": row["kind"], "question_id": row["question_id"],
                          **key, **result, "wall_seconds": wall}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
                m["successful_calls"] += 1
                if m["successful_calls"] % 50 == 0 or m["successful_calls"] == len(rows):
                    write_json(out / "manifest.json", m)
                    print(f"Jev continuation {m['task']}: {m['successful_calls']}/{len(rows)}", flush=True)
        jev.check(root, plan_dir)
        m.update(status="complete", results_sha256=frozen.sha256(out / "results.jsonl"))
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__,
                 failed_call_index=info["accepted_calls"] + m["attempted_additional_calls"] - 1)
        raise
    finally:
        m["usage"] = jev.usage_summary(records)
        write_json(out / "manifest.json", m)
    return out
