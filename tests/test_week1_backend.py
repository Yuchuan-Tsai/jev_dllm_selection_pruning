import copy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.backends.fast_llada import run_sampler, source_files, verify_sources
from jev_dllm.config import validate
from jev_dllm.evaluation.answers import numeric_answer, score_numeric
from jev_dllm.experiments.selection import evaluate_pools
from jev_dllm.experiments.backend_comparison import comparable_runs, compare
from jev_dllm.operations.dependencies import parse_lock, check_lock
from jev_dllm.schema import Candidate
from jev_dllm.selection.baselines import majority_numeric


class NumericVersionTests(unittest.TestCase):
    def test_v1_remains_conservative(self):
        self.assertFalse(score_numeric(r"\boxed{25\%}", "25"))
        self.assertIsNone(numeric_answer("Final answer: $18"))

    def test_v2_explicit_formats(self):
        cases = [(r"\boxed{25\%}", 25), ("#### 75 percent", 75),
                 (r"\boxed{\frac{3}{4}}", Fraction(3, 4)),
                 ("Final answer: $1,200.50", Fraction(2401, 2)),
                 (r"\boxed{12\text{ dollars}}", 12),
                 ("Final answer: 9 miles.", 9), ("Final answer: $8$", 8)]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(numeric_answer(text, "gsm8k_units_v2"), expected)
        self.assertNotEqual(numeric_answer("#### 25%", "gsm8k_units_v2"), Fraction(1, 4))

    def test_v2_does_not_guess_or_execute(self):
        for text in ("The answer is 25", "25", "#### 2 or 3", "#### 4 dollars and 5 cents",
                     "#### 1/0", r"\boxed{25", "#### __import__('os')", "#### 10 widgets"):
            with self.subTest(text=text):
                self.assertIsNone(numeric_answer(text, "gsm8k_units_v2"))
        with self.assertRaises(ValueError):
            numeric_answer("#### 4", "unknown")

    def test_voting_and_evaluation_use_same_version(self):
        rows = [Candidate("q", f"c{i}", i, text, True, False)
                for i, text in enumerate(["#### 7", "#### 9%", "#### 9 dollars"])]
        self.assertEqual(majority_numeric(rows), "c0")
        self.assertEqual(majority_numeric(rows, "gsm8k_units_v2"), "c1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q, c, r = [root / n for n in ("q.json", "c.jsonl", "r.json")]
            q.write_text(json.dumps([dict(question_id="q", dataset="gsm8k", split="dev", prompt="fixture")]))
            c.write_text("\n".join(json.dumps(asdict(x)) for x in rows))
            r.write_text(json.dumps([dict(question_id="q", answer="9")]))
            old = evaluate_pools(q, c, r, root / "runs")
            new = evaluate_pools(q, c, r, root / "runs", scorer="gsm8k_units_v2")
            self.assertNotEqual(old, new)
            self.assertEqual(json.loads((old / "metrics.json").read_text())["accuracy"]["majority_vote"], 0)
            metrics = json.loads((new / "metrics.json").read_text())
            self.assertEqual(metrics["accuracy"]["majority_vote"], 1)
            self.assertEqual(metrics["scorer_version"], "gsm8k_units_v2")


class FastBackendTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/week01/fast-diversity.json").read_text())

    def test_profile_pins_and_generation_settings(self):
        validate(self.config)
        original = json.loads((ROOT / "configs/week01/diversity.json").read_text())
        left, right = copy.deepcopy(original["generator"]), copy.deepcopy(self.config["generator"])
        left.pop("selected_backend"); right.pop("selected_backend")
        self.assertEqual(left, right)
        self.assertEqual(self.config["prompts"], original["prompts"])
        self.assertEqual(self.config["runtime"]["master_seed"], original["runtime"]["master_seed"])
        self.assertEqual(source_files(ROOT, self.config)["code_commit"], self.config["backends"]["fast_dllm_v1"]["code_commit"])

    def test_bad_threshold_and_source_pin_rejected(self):
        for value in (True, 0, 2, float("nan"), None):
            config = copy.deepcopy(self.config)
            config["backends"]["fast_dllm_v1"]["confidence_threshold"] = value
            with self.assertRaises(ValueError):
                validate(config)
        self.config["backends"]["fast_dllm_v1"]["source_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest"):
            source_files(ROOT, self.config)

    def test_modified_source_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(self.config)
            manifest = source_files(ROOT, config)
            for name, item in manifest["files"].items():
                path = root / "third_party/fast_dllm_v1" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("bad source")
            manifest_path = root / config["backends"]["fast_dllm_v1"]["source_manifest_path"]
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_bytes((ROOT / "configs/week01/fast-source.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "source hash"):
                verify_sources(root, config)

    def test_nfe_count_validation_and_hook_cleanup(self):
        model, handle = Mock(), Mock()
        model.register_forward_pre_hook.return_value = handle
        def sampler(model, prompt, **options):
            hook = model.register_forward_pre_hook.call_args.args[0]
            for _ in range(3):
                hook(model, ())
            return "output", 3
        options = dict(steps=4, gen_length=4, block_length=2, temperature=1., threshold=.9)
        self.assertEqual(run_sampler(model, sampler, "prompt", **options), ("output", 3))
        handle.remove.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "NFE"):
            run_sampler(model, lambda *a, **k: ("output", 1), "prompt", **options)

    def test_stalled_sampler_is_bounded(self):
        model, handle = Mock(), Mock()
        model.register_forward_pre_hook.return_value = handle
        def sampler(model, prompt, **options):
            hook = model.register_forward_pre_hook.call_args.args[0]
            for _ in range(5):
                hook(model, ())
        with self.assertRaisesRegex(RuntimeError, "bound"):
            run_sampler(model, sampler, "prompt", steps=4, gen_length=4, block_length=2, temperature=1., threshold=.9)
        handle.remove.assert_called_once()


class DependencyTests(unittest.TestCase):
    def test_exact_lock_syntax_and_bad_requirements(self):
        self.assertEqual(parse_lock(ROOT / "requirements/gpu-linux-py311.lock")["torch"], "2.5.1+cu121")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            for text in ("", "torch>=2", "torch @ https://example.com/pkg", "a==1\na==2"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    parse_lock(path)

    def test_mismatch_fails_without_installing(self):
        with patch("jev_dllm.operations.dependencies.platform.system", return_value="Linux"), \
             patch("jev_dllm.operations.dependencies.platform.machine", return_value="x86_64"), \
             patch("jev_dllm.operations.dependencies.sys.version_info", (3, 11)), \
             patch("jev_dllm.operations.dependencies.importlib.metadata.version", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "differs from lock"):
                check_lock(ROOT / "requirements/gpu-linux-py311.lock")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps([dict(question_id="q", dataset="gsm8k", split="dev", prompt="example")]))
        self.references = self.root / "references.json"
        self.references.write_text(json.dumps([dict(question_id="q", answer="5")]))
        self.paths = [self.root / "official", self.root / "fast"]
        for path, config_name in zip(self.paths, ("diversity.json", "fast-diversity.json")):
            path.mkdir()
            config = json.loads((ROOT / "configs/week01" / config_name).read_text())
            (path / "config.snapshot.json").write_text(json.dumps(config))
            (path / "manifest.json").write_text(json.dumps(dict(status="complete", completed_candidates=8,
                questions_sha256=hashlib.sha256(self.questions.read_bytes()).hexdigest(),
                total_seconds_including_load_and_warmup=10)))
            (path / "dependencies.txt").write_text("torch==fixture\n")
            (path / "candidates.jsonl").write_text("\n".join(json.dumps(asdict(Candidate("q", f"c{i}", i, "#### 5", True, False))) for i in range(8)))
            (path / "costs.jsonl").write_text("\n".join(json.dumps(dict(candidate_id=f"c{i}", question_id="q", trajectory_nfe=4,
                generation_gpu_seconds=.5, generation_wall_seconds=.6, peak_memory_bytes=100)) for i in range(8)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_matching_complete_runs_are_compared_offline(self):
        path = compare(*self.paths, self.questions, self.references, self.root / "outputs")
        result = json.loads((path / "comparison.json").read_text())
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["runs"]["fast_dllm_v1"]["nfe_total"], 32)
        self.assertEqual(result["scorer_version"], "gsm8k_units_v2")

    def test_changed_settings_or_environment_rejected(self):
        path = self.paths[1] / "config.snapshot.json"
        original = path.read_bytes()
        config = json.loads(original)
        config["generator"]["temperature"] = .7
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "settings"):
            comparable_runs(*self.paths, self.questions)
        path.write_bytes(original)
        (self.paths[1] / "dependencies.txt").write_text("torch==different")
        with self.assertRaisesRegex(ValueError, "dependency"):
            comparable_runs(*self.paths, self.questions)

    def test_partial_or_mismatched_cost_rows_rejected(self):
        path = self.paths[1] / "costs.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]))
        with self.assertRaisesRegex(ValueError, "Cost rows"):
            comparable_runs(*self.paths, self.questions)


if __name__ == "__main__":
    unittest.main()
