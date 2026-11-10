"""Finalize a full collection recovery that failed only at the old terminal-only source check."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_dllm.experiments import collection_recovery as recovery
from jev_dllm.experiments.selection import write_json
from jev_dllm.operations.remote import digest_file, snapshot_digest, verify_copy


def local_snapshot(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    files = {}
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError("Recovered run contains a symlink")
        if path.is_file():
            files[path.relative_to(run).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
            }
        elif not path.is_dir():
            raise ValueError("Recovered run contains an unexpected special file")
    snapshot = {"run_id": run.name, "status": manifest.get("status"), "files": files}
    verify_copy(run.parent, snapshot)
    return snapshot


def finalize(root: Path, source_run_id: str, *, confirm_error: bool) -> dict:
    if not confirm_error:
        raise ValueError("Confirm that the only reported failure was the terminal-only source verification error")
    plan_path = root / "artifacts/collection-recovery-plans" / f"{source_run_id}.json"
    plan, _, source_info = recovery.check_server_source(root, plan_path)
    if plan.get("task") != "math500" or plan.get("source_status") != "running":
        raise ValueError("Expected the stale-running MATH500 recovery plan")

    runs_root = root / "artifacts/runs"
    matching = []
    for run in runs_root.glob("fast-llada-math500-recovery-*"):
        manifest_path = run / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("recovery", {}).get("source_run_id") == source_run_id:
            matching.append((run, manifest))
    if len(matching) != 1:
        raise ValueError(f"Expected one recovery segment for {source_run_id}; found {len(matching)}")

    run, manifest = matching[0]
    if manifest.get("status") != "failed" or manifest.get("error_type") != "ValueError":
        raise ValueError("Only the recorded final source-check ValueError can be finalized")
    segment_info = recovery.inspect_run(root, run, snapshot=local_snapshot(run))
    total = len(segment_info["work"])
    if (segment_info["count"] != total
            or manifest.get("completed_candidates") != total
            or manifest.get("newly_generated_candidates") != total - plan["preserved_candidates"]
            or manifest.get("recovery", {}).get("preserved_candidates") != plan["preserved_candidates"]
            or source_info["count"] != plan["preserved_candidates"]):
        raise ValueError("Recovered candidate/cost counts or preserved prefix do not match the plan")

    manifest.pop("error_type", None)
    manifest["status"] = "complete"
    manifest["finalization"] = {
        "from_status": "failed",
        "from_error_type": "ValueError",
        "reason": "All candidates and costs passed recovery lineage checks; the source snapshot was reverified with the stale-running recovery check.",
        "source_run_id": source_run_id,
        "source_snapshot_digest": snapshot_digest(source_info["snapshot"]),
        "confirmed_terminal_only_error": True,
    }
    write_json(run / "manifest.json", manifest)
    return {"status": "complete", "run_id": run.name,
            "completed_candidates": total,
            "preserved_candidates": plan["preserved_candidates"],
            "newly_generated_candidates": total - plan["preserved_candidates"],
            "source_run_id": source_run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--confirm-terminal-only-error", action="store_true", required=True,
                        help="Confirm the reported error was the final terminal-only verification")
    args = parser.parse_args()
    try:
        print(json.dumps(finalize(args.root.resolve(), args.source_run_id,
                                  confirm_error=args.confirm_terminal_only_error), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Recovery finalization stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
