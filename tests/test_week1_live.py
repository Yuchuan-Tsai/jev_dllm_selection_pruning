"""Regression checks for live-run privacy, ProcessBench metrics, and safe smoke cleanup."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.cli import main
from jev_dllm.diagnostics.processbench import (CODE_REVISION, build_request, evaluate_subset,
                                               judge_subset, official_metrics)
from jev_dllm.experiments.smoke import run
from jev_dllm.judges.credentials import load_api_key
from jev_dllm.operations.smoke_cleanup import clean_successful_smokes, remove_successful_smoke


class CredentialsTests(unittest.TestCase):
    def test_only_key_loaded_without_executing_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / ".env"
            path.write_text("export TYPESAFE_API_KEY='fake-test-key'\nUNRELATED=do-not-load\n")
            load_api_key(path)
            self.assertEqual(os.environ["TYPESAFE_API_KEY"], "fake-test-key")
            self.assertNotIn("UNRELATED", os.environ)

    def test_existing_environment_key_wins(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "existing-test-key"}):
            load_api_key(Path("/does-not-exist"))
            self.assertEqual(os.environ["TYPESAFE_API_KEY"], "existing-test-key")

    def test_malformed_key_not_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / ".env"
            path.write_text("TYPESAFE_API_KEY='private-test-value\n")
            with self.assertRaises(ValueError) as error:
                load_api_key(path)
            self.assertNotIn("private-test-value", str(error.exception))


class SmokeCleanupTests(unittest.TestCase):
    def test_successful_cli_smoke_deletes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--root", str(ROOT), "smoke", "--output", tmp]), 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_keep_option_and_explicit_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            main(["--root", str(ROOT), "smoke", "--output", tmp, "--keep-artifacts"])
            path = next(Path(tmp).iterdir())
            summary = remove_successful_smoke(path, Path(tmp))
            self.assertGreater(summary["removed_bytes"], 0)
            self.assertFalse(path.exists())

    def test_failed_smoke_and_real_run_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "artifacts/runs"
            with patch("jev_dllm.experiments.selection.judge_pools", side_effect=RuntimeError("test failure")):
                with self.assertRaises(RuntimeError):
                    run(ROOT, parent)
            real = parent / "llada-dev-real"
            real.mkdir()
            (real / "candidates.jsonl").write_text("real-data")
            result = clean_successful_smokes(root)
            self.assertEqual(result["removed"], [])
            self.assertEqual(len(result["retained_for_inspection"]), 1)
            self.assertEqual((real / "candidates.jsonl").read_text(), "real-data")

    def test_legacy_success_recognized_from_terminal_cost_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = run(ROOT, Path(tmp))
            manifest = json.loads((path / "manifest.json").read_text())
            manifest.pop("status")
            (path / "manifest.json").write_text(json.dumps(manifest))
            remove_successful_smoke(path, Path(tmp))
            self.assertFalse(path.exists())

    def test_symlink_and_non_fixture_candidates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = run(ROOT, Path(tmp))
            link = Path(tmp) / "smoke-link"
            link.symlink_to(path, target_is_directory=True)
            with self.assertRaises(ValueError):
                remove_successful_smoke(link, Path(tmp))
            candidate = json.loads((path / "candidates.jsonl").read_text().splitlines()[0])
            candidate["question_id"] = "gsm8k/train/real"
            (path / "candidates.jsonl").write_text(json.dumps(candidate) + "\n")
            with self.assertRaises(ValueError):
                remove_successful_smoke(path, Path(tmp))
            self.assertTrue(path.exists())


class ProcessBenchTests(unittest.TestCase):
    def test_exact_first_error_and_correct_solution_harmonic_mean(self):
        result = official_metrics([0, 1, 2, -1, -1], [0, 0, 2, -1, 1])
        self.assertAlmostEqual(result["error_exact_index_accuracy_percent"], 200 / 3)
        self.assertEqual(result["correct_solution_accuracy_percent"], 50)
        self.assertAlmostEqual(result["f1_percent"], 400 / 7)
        # Detecting an error at the wrong index receives no credit.
        self.assertEqual(official_metrics([1, -1], [0, 0])["f1_percent"], 0)
        self.assertIsNone(official_metrics([0], [0])["f1_percent"])
        self.assertEqual(official_metrics([0, -1], [None, -1])["f1_percent"], 0)

    def test_payload_excludes_private_labels_and_validates_steps(self):
        row = {"question_id": "diagnostic", "subset": "gsm8k", "problem": "a problem", "steps": ["a step", "another"]}
        request = build_request(row, "Find earliest error", "jev-1.13.0")
        self.assertEqual(set(request["state"]), {"problem", "steps"})
        self.assertEqual(set(request["questions"]["best_answer"]["criteria"]), {"-1", "0", "1"})
        for bad in ({**row, "label": 0}, {**row, "final_answer_correct": False}, {**row, "steps": []}):
            with self.assertRaises(ValueError):
                build_request(bad, "Find earliest error", "jev-1.13.0")
        with self.assertRaises(ValueError):
            build_request(row, "Find earliest error", "jev-latest")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data_manifest = {"official_code_revision": CODE_REVISION, "subsets": {}, "files": {}}
        for subset in ("gsm8k", "math"):
            rows = [{"question_id": subset + str(i), "subset": subset, "problem": f"problem {i}", "steps": ["step"]} for i in range(2)]
            labels = [{"question_id": r["question_id"], "label": -1 if i == 0 else 0} for i, r in enumerate(rows)]
            for rel, value in [(f"data/processed/processbench-{subset}-diagnostic.json", rows),
                               (f"data/private/processbench-{subset}-labels.json", labels)]:
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                data_manifest["files"][rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            data_manifest["subsets"][subset] = {"question_ids": [r["question_id"] for r in rows]}
        manifest = self.root / "data/manifests/processbench-week01.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(data_manifest))
        prompt = self.root / "prompt.txt"
        prompt.write_text("Select earliest erroneous step or -1.")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"model": "jev-1.13.0", "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "prompt": {"path": "prompt.txt", "sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()}}))

    def test_dry_run_does_not_read_labels_or_call_network(self):
        for p in (self.root / "data/private").iterdir():
            p.unlink()
        with patch("jev_dllm.judges.typesafe.TypeSafeJudge.select") as select:
            result = judge_subset(self.root, self.root / "runs", self.config)
            select.assert_not_called()
        manifest = json.loads((result / "manifest.json").read_text())
        self.assertEqual(manifest["attempted_calls"], 0)
        self.assertFalse((result / "predictions.jsonl").exists())
        with self.assertRaises(ValueError):
            evaluate_subset(self.root, result, self.root / "runs")

    def test_completed_fixture_scoring_and_tamper_rejection(self):
        class FakeJudge:
            def select(self, request):
                return {"candidate_id": "-1"}
        result = judge_subset(self.root, self.root / "runs", self.config, live=True, judge=FakeJudge())
        evaluated = evaluate_subset(self.root, result, self.root / "runs")
        scores = json.loads((evaluated / "metrics.json").read_text())
        self.assertEqual(scores["subset_results"]["gsm8k"]["correct_solution_accuracy_percent"], 100)
        self.assertEqual(scores["subset_results"]["gsm8k"]["f1_percent"], 0)
        with (result / "predictions.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaises(ValueError):
            evaluate_subset(self.root, result, self.root / "runs")

    def test_failure_retains_partial_run_without_retry(self):
        class FailingJudge:
            count = 0
            def select(self, request):
                self.count += 1
                raise RuntimeError("service failed")
        judge = FailingJudge()
        with self.assertRaises(RuntimeError):
            judge_subset(self.root, self.root / "runs", self.config, live=True, judge=judge)
        self.assertEqual(judge.count, 1)
        path = next((self.root / "runs").iterdir())
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["attempted_calls"], 1)
        with self.assertRaises(ValueError):
            evaluate_subset(self.root, path, self.root / "runs")

    def test_nonfinite_rejected_response_does_not_break_failure_audit(self):
        class InvalidJudge:
            def select(self, request):
                error = ValueError("invalid probability")
                error.raw_response = {"probabilities": {"0": float("nan")}}
                error.request_seconds = .2
                raise error
        with self.assertRaises(ValueError):
            judge_subset(self.root, self.root / "runs", self.config, live=True, judge=InvalidJudge())
        run_dir = next((self.root / "runs").iterdir())
        self.assertEqual(json.loads((run_dir / "manifest.json").read_text())["status"], "failed")
        self.assertIn("NaN", json.loads((run_dir / "failed_response.json").read_text())["raw_response_json"])

    def test_tampered_inputs_rejected_before_calls(self):
        path = self.root / "data/processed/processbench-gsm8k-diagnostic.json"
        path.write_text("[]")
        with patch("jev_dllm.judges.typesafe.TypeSafeJudge.select") as select:
            with self.assertRaises(ValueError):
                judge_subset(self.root, self.root / "runs", self.config, live=True)
            select.assert_not_called()


if __name__ == "__main__":
    unittest.main()
