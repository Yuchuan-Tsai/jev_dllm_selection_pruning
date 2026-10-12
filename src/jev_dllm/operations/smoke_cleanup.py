"""Remove only validated, successful synthetic smoke artifacts."""
import json
import math
from pathlib import Path
import shutil


def remove_successful_smoke(run: Path, parent: Path) -> dict:
    if run.is_symlink() or not run.name.startswith("smoke-") or run.resolve().parent != parent.resolve():
        raise ValueError("Cleanup is limited to a direct synthetic smoke directory")
    paths = list(run.rglob("*"))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
        raise ValueError("Smoke cleanup rejects symlinks and special files")
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest.get("mode") != "synthetic_smoke_only" or manifest.get("is_benchmark_result") is not False
            or manifest.get("generator") is not None or manifest.get("judge_called") is not False
            or manifest.get("status") not in (None, "complete")):
        raise ValueError("Only completed synthetic, model-free, API-free smoke results can be removed")
    # Legacy smoke versions wrote costs.json only after all checks succeeded.
    costs = json.loads((run / "costs.json").read_text())
    elapsed = costs.get("end_to_end_seconds")
    if (costs.get("mode") != "synthetic_smoke_only" or costs.get("api_cost_usd") != 0
            or costs.get("generation_gpu_seconds") is not None
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Successful synthetic completion evidence is missing")
    rows = [json.loads(line) for line in (run / "candidates.jsonl").read_text().splitlines()]
    if not rows or any(not row["question_id"].startswith("fixture-") for row in rows):
        raise ValueError("Refusing to remove non-fixture candidate data")
    allowed = {"manifest.json", "metrics.json", "costs.json", "config.snapshot.json",
               "questions.json", "candidates.jsonl", "judge_payloads.jsonl",
               "evaluation_private", "selection_checks"}
    if any(p.name not in allowed for p in run.iterdir()):
        raise ValueError("Unexpected files in smoke directory; retain for manual inspection")
    for p in (run / "selection_checks").glob("*/manifest.json"):
        nested = json.loads(p.read_text())
        if nested.get("mode") not in {"offline_evaluation", "dry_run"} or nested.get("attempted_calls", 0) != 0:
            raise ValueError("Refusing to remove a smoke directory containing live activity")
    summary = {"run_id": run.name, "removed_bytes": sum(p.stat().st_size for p in paths if p.is_file()),
               "mode": "synthetic_smoke_only"}
    shutil.rmtree(run)
    return summary


def clean_successful_smokes(root: Path) -> dict:
    parent = root / "artifacts/runs"
    removed, retained = [], []
    for run in sorted(parent.glob("smoke-*")):
        try:
            removed.append(remove_successful_smoke(run, parent))
        except (OSError, ValueError, KeyError, TypeError):
            retained.append(run.name)
    return {"removed": removed, "retained_for_inspection": retained,
            "removed_bytes": sum(r["removed_bytes"] for r in removed)}
