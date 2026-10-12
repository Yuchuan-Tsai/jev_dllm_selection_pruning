from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.config import load, validate
from jev_dllm.cli import main
from jev_dllm.backends.official_llada import OfficialLLaDA
from jev_dllm.data.gsm8k import prepare
from jev_dllm.experiments.selection import evaluate_pools, judge_pools, paired_summary
from jev_dllm.judges.typesafe import TypeSafeJudge, build_request, parse_response
from jev_dllm.schema import Candidate, Question
from jev_dllm.selection.baselines import majority_numeric, random_choice


class SelectionTests(unittest.TestCase):
    def test_majority_normalizes_fractions_and_ignores_invalid_votes(self):
        rows = [Candidate("q", f"c{i}", i, text, True, False) for i, text in enumerate(
            ["Final answer: 7", "Final answer: 0.5", r"\boxed{\frac{1}{2}}", "5 or 6"])]
        self.assertEqual(majority_numeric(rows), "c1")

    def test_ties_fallback_and_truncation(self):
        a = Candidate("q", "a", 1, "Final answer: 2", True, False)
        b = Candidate("q", "b", 2, "Final answer: 3", True, False)
        self.assertEqual(majority_numeric([a, b]), "a")
        self.assertEqual(majority_numeric([replace(a, truncated=True), b]), "b")
        self.assertEqual(majority_numeric([replace(a, text="invalid"), replace(b, text="invalid")]), "a")

    def test_random_is_reproducible_and_pools_are_validated(self):
        rows = [Candidate("q", str(i), i, "#### 5", True, False) for i in range(8)]
        self.assertEqual(random_choice(rows, 123), random_choice(rows, 123))
        with self.assertRaises(ValueError):
            majority_numeric([rows[0], rows[0]])


class JevTests(unittest.TestCase):
    def setUp(self):
        self.q = Question("q", "gsm8k", "dev", "Compute 2+3")
        self.candidates = [Candidate("q", "a", 1, "#### 5", True, False), Candidate("q", "b", 2, "#### 6", True, False)]
        self.request = build_request(self.q, self.candidates, "Choose the correct candidate.")
        self.response = {"model": "jev-1.13.0", "answers": {"best_answer": {
            "type": "choice", "choice": "a", "probabilities": {"a": .8, "b": .2}, "confidence": .6}},
            "usage": {"input_tokens": 12, "output_tokens": 3}}

    def test_payload_has_no_private_or_internal_fields(self):
        body = self.request
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(set(body["state"]), {"stage", "question", "candidates"})
        self.assertEqual(set(body["state"]["candidates"][0]), {"candidate_id", "text"})
        for forbidden in ("reference", "correctness", "seed", "confidence"):
            self.assertNotIn(forbidden, json.dumps(body))

    def test_aliases_incomplete_and_one_candidate_rejected(self):
        for pool, model in [(self.candidates, "jev-latest"), (self.candidates[:1], "jev-1.13.0"),
                            ([replace(self.candidates[0], truncated=True), self.candidates[1]], "jev-1.13.0")]:
            with self.assertRaises(ValueError):
                build_request(self.q, pool, "select", model)

    def test_valid_response(self):
        result = parse_response(self.response, self.request)
        self.assertEqual(result["candidate_id"], "a")
        self.assertEqual(result["usage"]["input_tokens"], 12)

    def test_wrong_model_choice_or_probabilities_rejected(self):
        changes = [{"choice": "unknown"}, {"probabilities": {"a": 1}},
                   {"probabilities": {"a": float("nan"), "b": .2}},
                   {"probabilities": {"a": .8, "b": .8}}, {"choice": "b"}, {"confidence": True}]
        for change in changes:
            response = copy.deepcopy(self.response)
            response["answers"]["best_answer"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_response(response, self.request)
        self.response["model"] = "jev-latest"
        with self.assertRaises(ValueError):
            parse_response(self.response, self.request)

    def test_two_decimal_rounding_is_bounded_and_not_normalized(self):
        response = copy.deepcopy(self.response)
        response["answers"]["best_answer"]["probabilities"] = {"a": .80, "b": .19}
        result = parse_response(response, self.request)
        self.assertAlmostEqual(result["probability_sum"], .99)
        self.assertEqual(result["probabilities"], {"a": .80, "b": .19})
        for values in ({"a": .80, "b": .17}, {"a": 0, "b": 0}, {"a": .801, "b": .19}):
            response["answers"]["best_answer"]["probabilities"] = values
            with self.assertRaises(ValueError):
                parse_response(response, self.request)

    def test_invalid_live_response_is_preserved_on_exception(self):
        response = copy.deepcopy(self.response)
        response["answers"]["best_answer"]["probabilities"] = {"a": .8, "b": .8}
        from unittest.mock import MagicMock
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = io.StringIO(json.dumps(response))
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "fake-test-key"}), patch(
                "jev_dllm.judges.typesafe.build_opener", return_value=opener):
            with self.assertRaises(ValueError) as raised:
                TypeSafeJudge().select(self.request)
        self.assertEqual(raised.exception.raw_response, response)
        self.assertGreaterEqual(raised.exception.request_seconds, 0)

    def test_missing_key_never_opens_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("jev_dllm.judges.typesafe.build_opener") as opener:
            with self.assertRaises(ValueError):
                TypeSafeJudge().select(self.request)
            opener.assert_not_called()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.q = self.root / "questions.json"
        self.c = self.root / "candidates.jsonl"
        self.r = self.root / "references.json"
        self.prompt = self.root / "prompt.txt"
        self.q.write_text(json.dumps([asdict(Question("q", "synthetic", "fixture", "2+3?"))]))
        rows = [Candidate("q", "a", 1, "#### 6", True, False), Candidate("q", "b", 2, "#### 5", True, False)]
        self.c.write_text("".join(json.dumps(asdict(row)) + "\n" for row in rows))
        self.r.write_text(json.dumps([{"question_id": "q", "answer": "5"}]))
        self.prompt.write_text("Choose the correct answer.")

    def test_dry_run_never_calls_jev_or_fabricates_scores(self):
        with patch("jev_dllm.judges.typesafe.TypeSafeJudge.select") as call:
            path = judge_pools(self.q, self.c, self.prompt, self.root, "jev-1.13.0")
            call.assert_not_called()
        self.assertFalse((path / "selections.jsonl").exists())
        self.assertEqual(json.loads((path / "manifest.json").read_text())["attempted_calls"], 0)
        path = evaluate_pools(self.q, self.c, self.r, self.root)
        stats = json.loads((path / "metrics.json").read_text())
        self.assertFalse(stats["jev_evaluated"])
        self.assertNotIn("jev", stats["accuracy"])
        self.assertEqual(stats["expected_uniform_random_accuracy"], .5)
        self.assertEqual(stats["empirical_oracle_at_pool_size"], 1)

    def test_mocked_judging_and_paired_comparison(self):
        class FakeJudge:
            def select(self, request):
                return {"candidate_id": "b", "api_cost_usd": None}
        path = judge_pools(self.q, self.c, self.prompt, self.root, "jev-1.13.0", live=True, judge=FakeJudge())
        result = evaluate_pools(self.q, self.c, self.r, self.root, path / "selections.jsonl")
        stats = json.loads((result / "metrics.json").read_text())
        self.assertEqual(stats["accuracy"]["jev"], 1)
        self.assertEqual(stats["jev_vs_majority"]["wins"], 1)
        # Same IDs but changed text cannot reuse an old judging result.
        self.c.write_text(self.c.read_text().replace("#### 6", "#### 7"))
        with self.assertRaises(ValueError):
            evaluate_pools(self.q, self.c, self.r, self.root, path / "selections.jsonl")

    def test_failures_preserve_an_audit_without_fallback(self):
        class FailedJudge:
            def select(self, request):
                raise RuntimeError("simulated service failure")
        with self.assertRaises(RuntimeError):
            judge_pools(self.q, self.c, self.prompt, self.root, "jev-1.13.0", live=True, judge=FailedJudge())
        manifest = json.loads(next(self.root.glob("jev-live-*/manifest.json")).read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["successful_calls"], 0)
        self.assertIsNone(manifest["api_cost_usd"])

    def test_missing_candidates_or_references_cannot_reduce_denominator(self):
        for path in (self.q, self.r):
            before = path.read_text()
            path.write_text("[]")
            with self.assertRaises(ValueError):
                evaluate_pools(self.q, self.c, self.r, self.root)
            path.write_text(before)

    def test_invalid_candidates_count_as_failures_in_offline_summary(self):
        rows = [json.loads(line) for line in self.c.read_text().splitlines()]
        rows[1]["truncated"] = True
        self.c.write_text("".join(json.dumps(row) + "\n" for row in rows))
        path = evaluate_pools(self.q, self.c, self.r, self.root)
        stats = json.loads((path / "metrics.json").read_text())
        self.assertEqual(stats["expected_uniform_random_accuracy"], 0)
        self.assertEqual(stats["incomplete_or_truncated_fraction"], .5)
        with self.assertRaises(ValueError):
            judge_pools(self.q, self.c, self.prompt, self.root, "jev-1.13.0")


class DataAndConfigTests(unittest.TestCase):
    def test_preparation_is_reproducible_and_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "test"):
                (root / f"{split}.jsonl").write_text("".join(json.dumps({"question": f"{split} {i}?", "answer": f"Work. #### {i}"}) + "\n" for i in range(5)))
            args = (root, root / "train.jsonl", root / "test.jsonl", "a"*40, 2, 3)
            path = prepare(*args)
            before = path.read_bytes()
            prepare(*args)
            self.assertEqual(before, path.read_bytes())
            public = json.loads((root / "data/processed/gsm8k-dev.json").read_text())
            self.assertNotIn("answer", public[0])
            self.assertEqual(len(json.loads((root / "data/private/gsm8k-heldout-references.json").read_text())), 3)

    def test_pilot_is_not_frozen_or_claimed_ready(self):
        config, pending = load(ROOT / "configs/week01/pilot.json")
        self.assertIn("runtime.hardware", pending)
        self.assertIn("readiness.jev_access_validated", pending)
        config["frozen"] = True
        with self.assertRaises(ValueError):
            validate(config)

    def test_pilot_rejects_incompatible_schedule_and_mutable_model_pin(self):
        config, _ = load(ROOT / "configs/week01/pilot.json")
        for section, key, value in [("generator", "temperature", 0), ("generator", "model_revision", "main"),
                                    ("generator", "candidates_per_question", True)]:
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate(changed)
        config["backends"]["official_llada"]["steps"] = 7
        with self.assertRaises(ValueError):
            validate(config)

    def test_paired_intervals_use_question_units(self):
        result = paired_summary([True, False], [False, True], 42)
        self.assertEqual(result["accuracy_difference"], 0)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        with self.assertRaises(ValueError):
            paired_summary([True], [False, True], 42)

    def test_generator_rejects_unpinned_inputs_before_loading_dependencies(self):
        config, _ = load(ROOT / "configs/week01/pilot.json")
        config["generator"]["model_revision"] = None
        with self.assertRaisesRegex(ValueError, "Immutable"):
            OfficialLLaDA(config, Path("missing.py"), "{question}")
        config, _ = load(ROOT / "configs/week01/pilot.json")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            OfficialLLaDA(config, Path("missing.py"), "Question was accidentally omitted")

    def test_cli_blocks_heldout_outcomes_before_reading_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            questions = Path(tmp) / "questions.json"
            questions.write_text(json.dumps([asdict(Question("q", "gsm8k", "heldout", "a question"))]))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                code = main(["--root", str(ROOT), "evaluate-pool", "--questions", str(questions),
                             "--candidates", "missing-candidates", "--references", "missing-references"])
            self.assertEqual(code, 2)
            self.assertIn("Freeze", err.getvalue())


if __name__ == "__main__":
    unittest.main()
