"""Independent fixtures for the opt-in compound-unit policy; no model/API calls."""
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.evaluation.answers import numeric_answer
from jev_dllm.experiments.selection import evaluate_pools
from jev_dllm.schema import Candidate
from jev_dllm.selection.baselines import majority_numeric


class CompoundUnitPolicyTests(unittest.TestCase):
    def test_explicit_compound_unit_formats(self):
        examples = [
            (r"\boxed{92 \text{ seconds/day}}", Fraction(92)),
            (r"\boxed{7.5\mathrm{miles/hour}}", Fraction(15, 2)),
            ("#### 18 dollars per hour", Fraction(18)),
            ("Final answer: 6 items/week.", Fraction(6)),
            ("#### 2,400 meters / minute", Fraction(2400)),
            ("#### -3 pounds per day", Fraction(-3)),
            (r"\boxed{\frac{3}{4} \text{miles/hour}}", Fraction(3, 4)),
            ("Final answer: 1/2 kilometers/hour", Fraction(1, 2)),
            ("#### 12 MINUTES PER DAY", Fraction(12)),
        ]
        for text, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(numeric_answer(text, "gsm8k_units_v3"), expected)

    def test_older_versions_and_default_remain_unchanged(self):
        example = r"\boxed{92 \text{seconds/day}}"
        for scorer in ("gsm8k_strict_v1", "gsm8k_units_v2"):
            self.assertIsNone(numeric_answer(example, scorer))
        self.assertIsNone(numeric_answer(example))
        self.assertIsNone(numeric_answer(r"\boxed{27\%}"))
        self.assertEqual(numeric_answer(r"\boxed{27\%}", "gsm8k_units_v2"), 27)

    def test_no_unit_conversion_or_reference_dependent_guessing(self):
        self.assertEqual(numeric_answer("#### 2 hours/day", "gsm8k_units_v3"), 2)
        self.assertNotEqual(numeric_answer("#### 2 hours/day", "gsm8k_units_v3"), 120)
        self.assertEqual(numeric_answer("#### 27%", "gsm8k_units_v3"), 27)
        self.assertNotEqual(numeric_answer("#### 27%", "gsm8k_units_v3"), Fraction(27, 100))
        self.assertNotEqual(numeric_answer("#### 17 miles/hour", "gsm8k_units_v3"), 18)

    def test_unsupported_ambiguous_and_unmarked_text_is_rejected(self):
        examples = [
            "The answer is 92 seconds/day", "92 seconds/day", r"\boxed{}92 seconds/day",
            "#### 12 minutes/week or 14 minutes/week", "#### 4 dollars and 5 cents",
            "#### 9 miles/hour/day", "#### 10 widgets/hour", "#### 10 meters/kilogram",
            "#### 15 / week", "#### 12 mph", "#### 12 dollarsperhour",
            "#### 2 + 3 miles/hour", "#### 1/0 miles/hour", "#### nan miles/hour",
            "#### __import__('os')", r"\boxed{12\text{miles/hour}",
            r"#### 12\text{miles/hour} extra", "#### 12 miles/hour " + "x" * 512,
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertIsNone(numeric_answer(text, "gsm8k_units_v3"))

    def test_scalar_formats_are_preserved(self):
        for text in ("#### 42", "#### 14 percent", "#### $1,025.50", r"\boxed{9\text{ dollars}}",
                     r"\boxed{\frac{2}{3}}", "Final answer: 5 miles."):
            with self.subTest(text=text):
                self.assertEqual(numeric_answer(text, "gsm8k_units_v3"), numeric_answer(text, "gsm8k_units_v2"))

    def test_final_marker_precedence_remains_conservative(self):
        self.assertEqual(numeric_answer(r"\boxed{3 miles/hour} then \boxed{8 miles/hour}", "gsm8k_units_v3"), 8)
        self.assertIsNone(numeric_answer(r"#### 8 miles/hour then \boxed{}", "gsm8k_units_v3"))

    def test_voting_and_evaluation_share_policy_and_preserve_outputs(self):
        candidates = [Candidate("fixture", f"c{i}", i, text, True, False)
                      for i, text in enumerate(["#### 4", "#### 9 meters/minute", "#### 9 meters per minute"])]
        self.assertEqual(majority_numeric(candidates, "gsm8k_units_v2"), "c0")
        self.assertEqual(majority_numeric(candidates, "gsm8k_units_v3"), "c1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions, pool, refs = [root / name for name in ("questions.json", "candidates.jsonl", "refs.json")]
            questions.write_text(json.dumps([dict(question_id="fixture", dataset="gsm8k", split="dev", prompt="synthetic")]))
            pool.write_text("\n".join(json.dumps(asdict(c)) for c in candidates))
            refs.write_text(json.dumps([dict(question_id="fixture", answer="9")]))
            before = pool.read_bytes()
            old = evaluate_pools(questions, pool, refs, root / "runs", scorer="gsm8k_units_v2")
            old_bytes = (old / "metrics.json").read_bytes()
            new = evaluate_pools(questions, pool, refs, root / "runs", scorer="gsm8k_units_v3")
            self.assertNotEqual(old, new)
            self.assertEqual((old / "metrics.json").read_bytes(), old_bytes)
            self.assertEqual(pool.read_bytes(), before)
            result = json.loads((new / "metrics.json").read_text())
            self.assertEqual(result["accuracy"]["majority_vote"], 1)
            self.assertEqual(result["mean_single_sample_accuracy"], 2/3)
            self.assertEqual(result["scorer_version"], "gsm8k_units_v3")


if __name__ == "__main__":
    unittest.main()
