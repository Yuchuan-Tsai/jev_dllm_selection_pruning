"""Synthetic candidates only. This is not a generator or benchmark experiment."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import subprocess
import time
import uuid
from ..evaluation.answers import score_numeric
from ..evaluation.metrics import pool_metrics
from ..judges.payloads import final_request
from ..reproducibility import fingerprint, trajectory_seed
from ..schema import Candidate, CostRecord, Question


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(root: Path, output: Path | None = None) -> Path:
    started = time.perf_counter()
    fixture_path = root / "tests/fixtures/smoke_questions.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("smoke-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    run_dir = (output or root / "artifacts/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    public_rows, private_rows, requests = [], [], []
    texts, labels = {}, {}
    for item in fixtures:
        question = Question(item["question_id"], "synthetic_numeric", "fixture", item["prompt"])
        candidates = [Candidate(question.question_id, f"{question.question_id}/c{i:02}",
                                trajectory_seed(20261008, question.question_id, i), text, True, False)
                      for i, text in enumerate(item["candidate_texts"])]
        public_rows.extend(asdict(c) for c in candidates)
        requests.append(final_request(question, candidates))
        texts[question.question_id] = [c.text for c in candidates]
        labels[question.question_id] = [score_numeric(c.text, item["reference"]) for c in candidates]
        private_rows.extend({"question_id": c.question_id, "candidate_id": c.candidate_id,
                             "correct": correct, "reference": item["reference"]}
                            for c, correct in zip(candidates, labels[question.question_id]))
    for name, rows in (("candidates.jsonl", public_rows), ("judge_payloads.jsonl", requests)):
        (run_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (run_dir / "evaluation_private").mkdir()
    _write_json(run_dir / "evaluation_private/labels.json", private_rows)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=False, timeout=5).stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        commit = None
    config = json.loads((root / "configs/week01/protocol.json").read_text(encoding="utf-8"))
    _write_json(run_dir / "config.snapshot.json", config)
    _write_json(run_dir / "manifest.json", {
        "run_id": run_id, "mode": "synthetic_smoke_only", "is_benchmark_result": False,
        "generator": None, "judge_called": False, "git_commit": commit,
        "python": platform.python_version(), "platform": platform.platform(),
        "fixture_sha256": fingerprint(fixtures), "protocol_sha256": fingerprint(config),
        "seed_policy": "sha256(master_seed, question_id, candidate_index)",
        "master_seed": 20261008,
    })
    _write_json(run_dir / "metrics.json", {"mode": "synthetic_smoke_only", **pool_metrics(texts, labels)})
    cost = asdict(CostRecord(end_to_end_seconds=time.perf_counter() - started, api_cost_usd=0.0))
    _write_json(run_dir / "costs.json", {"mode": "synthetic_smoke_only", **cost})
    return run_dir
