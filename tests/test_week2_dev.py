"""Week 2 development-set sampling, overlap guards and MBPP conversion; no network."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.data import week2_dev

MBPP_GOOD = {"task_id": 11, "prompt": "Write a function to add two numbers.",
             "code": "def add(a, b):\n    return a + b", "test_imports": [],
             "test_list": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]}


def mbpp_row(task_id, **changes):
    return {**MBPP_GOOD, "task_id": task_id, "prompt": f"Task {task_id}: write a function to add two numbers.", **changes}


class Week2DevTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        gsm = [{"question": f"Question {i}?", "answer": f"Work\n#### {i}"} for i in range(8)]
        gsm[5]["question"] = "Held-out duplicate?"
        math_rows = [{"problem": f"Problem {i}", "answer": str(i), "subject": "Algebra", "level": 1,
                      "unique_id": f"train/algebra/{i}.json", "solution": ""} for i in range(6)]
        math_rows += [{**math_rows[0], "unique_id": "test/algebra/9.json", "problem": "Test-split problem"},
                      {**math_rows[1], "unique_id": "train/algebra/1.json", "problem": "Duplicate ID"},
                      {**math_rows[2], "unique_id": "train/algebra/50.json", "answer": " ", "problem": "Empty answer"}]
        mbpp = [mbpp_row(4)] + [mbpp_row(i) for i in range(11, 17)]
        mbpp.append(mbpp_row(20, code="def helper(x):\n    return x\ndef add(a, b):\n    return a + b",
                             test_list=["assert add(helper(1), 2) == 3"]))
        self.sources = {
            "gsm8k": self.raw("data/raw/gsm8k-train.jsonl", "".join(json.dumps(r) + "\n" for r in gsm), len(gsm), "jsonl"),
            "math": self.raw("data/raw/prm800k-math-train.jsonl", "".join(json.dumps(r) + "\n" for r in math_rows), len(math_rows), "jsonl"),
            "mbpp": self.raw("data/raw/mbpp-sanitized.json", json.dumps(mbpp), len(mbpp), "json"),
        }
        protected = {"data/processed/gsm8k-dev.json": [{"question_id": "gsm8k/train/00000", "prompt": "Question 0?"}],
                     "data/processed/gsm8k-heldout.json": [{"question_id": "h", "prompt": "Held-out  duplicate?"}],
                     "data/processed/processbench-gsm8k-diagnostic.json": [{"question_id": "p", "problem": "Problem 3"}]}
        for relative, rows in protected.items():
            self.write(relative, json.dumps(rows))
        self.catalog = {"schema_version": 1, "scope": "week02_development_pools", "seed": 7, "count_per_task": 3,
                        "tasks": {"gsm8k": {"repository": "r", "revision": "a" * 40, "source": self.sources["gsm8k"]},
                                  "math": {"repository": "r", "revision": "b" * 40, "source": self.sources["math"],
                                           "unique_id_prefix": "train/"},
                                  "mbpp": {"repository": "r", "revision": "c" * 40, "source": self.sources["mbpp"],
                                           "min_task_id": 11}},
                        "protected_public_inputs": sorted(protected)}
        self.write(week2_dev.CATALOG, json.dumps(self.catalog))

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def raw(self, relative, text, rows, fmt):
        self.write(relative, text)
        data = text.encode()
        return {"path": relative, "url": "https://example.invalid/" + relative, "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data), "rows": rows, "format": fmt}

    def read(self, relative):
        return json.loads((self.root / relative).read_text())

    def test_prepare_is_deterministic_disjoint_and_refuses_changes(self):
        summary = week2_dev.prepare(self.root, self.catalog)
        self.assertEqual({t: s["questions"] for t, s in summary.items()}, {"gsm8k": 3, "math": 3, "mbpp": 3})
        first = {p: (self.root / p).read_bytes() for p in json.loads((self.root / week2_dev.MANIFEST).read_text())["files"]}
        week2_dev.prepare(self.root, self.catalog)  # Identical rerun is accepted unchanged.
        for path, raw in first.items():
            self.assertEqual((self.root / path).read_bytes(), raw)
        gsm = self.read("data/processed/gsm8k-w2dev.json")
        self.assertTrue(all(q["split"] == "dev" and q["dataset"] == "gsm8k" for q in gsm))
        self.assertNotIn("gsm8k/train/00000", {q["question_id"] for q in gsm})  # Pilot development question.
        self.assertNotIn("Held-out duplicate?", {q["prompt"] for q in gsm})
        refs = self.read("data/private/gsm8k-w2dev-references.json")
        self.assertEqual([r["question_id"] for r in refs], [q["question_id"] for q in gsm])
        self.assertTrue(all("answer" not in q for q in gsm))
        math_ids = {q["question_id"] for q in self.read("data/processed/math-w2dev.json")}
        self.assertFalse(math_ids & {"math/test/algebra/9.json", "math/train/algebra/1.json", "math/train/algebra/50.json"})
        self.assertNotIn("math/train/algebra/3.json", math_ids)  # Matches a ProcessBench problem.
        changed = dict(self.catalog, seed=8)
        self.write(week2_dev.CATALOG, json.dumps(changed))
        with self.assertRaises(ValueError):
            week2_dev.audit(self.root, changed)
        with self.assertRaises(ValueError):
            week2_dev.prepare(self.root, changed)

    def test_mbpp_conversion_excludes_prompt_split_and_ambiguous_tasks(self):
        week2_dev.prepare(self.root, self.catalog)
        manifest = json.loads((self.root / week2_dev.MANIFEST).read_text())
        mbpp = self.read("data/processed/mbpp-w2dev.json")
        self.assertNotIn("mbpp/4", {q["question_id"] for q in mbpp})
        self.assertIn({"question_id": "mbpp/20", "reason": "ambiguous_or_missing_entry_point"},
                      manifest["tasks"]["mbpp"]["dropped_during_conversion"])
        for question_id, source, reference in week2_dev.mbpp_reference_checks(self.root):
            self.assertTrue(question_id.startswith("mbpp/"))
            self.assertEqual(source, reference["canonical_solution"])

    def test_converted_prompt_and_check_behave_like_humaneval(self):
        prompt, reference = week2_dev.convert_mbpp(MBPP_GOOD)
        self.assertTrue(prompt.startswith("def add(a, b):\n"))
        self.assertIn("Example:\n    assert add(1, 2) == 3", prompt)
        compile(prompt + "    pass\n", "<prompt>", "exec")
        # Fixture code only: verify the converted check accepts a correct and rejects a wrong solution.
        for source, passes in (("def add(a, b):\n    return a + b", True), ("def add(a, b):\n    return a - b", False)):
            namespace = {}
            exec(source, namespace)
            exec(reference["hidden_tests"], namespace)
            if passes:
                namespace["check"](namespace["add"])
            else:
                with self.assertRaises(AssertionError):
                    namespace["check"](namespace["add"])

    def test_test_imports_and_backslashes_are_preserved(self):
        row = dict(MBPP_GOOD, code="import re\ndef digits(s):\n    return re.findall(r'\\d', s)",
                   test_imports=["import re"], test_list=["assert digits('a1b2') == re.findall(r'\\d', 'a1b2')"])
        prompt, reference = week2_dev.convert_mbpp(row)
        self.assertIn('r"""', prompt)
        self.assertIn("r'\\d'", prompt)
        self.assertIn("    import re\n    digits = candidate\n", reference["hidden_tests"])
        namespace = {}
        exec(row["code"], namespace)
        exec(reference["hidden_tests"], namespace)
        namespace["check"](namespace["digits"])

    def test_tests_needing_reference_only_definitions_are_rejected(self):
        row = dict(MBPP_GOOD, code="class Box:\n    pass\ndef add(a, b):\n    return a + b",
                   test_list=["assert add(1, 2) == 3 and Box is not None"])
        with self.assertRaisesRegex(week2_dev.ConversionError, "other_reference_definitions"):
            week2_dev.convert_mbpp(row)

    def test_missing_protected_input_or_too_few_rows_stop(self):
        too_many = dict(self.catalog, count_per_task=50)
        with self.assertRaisesRegex(ValueError, "Not enough eligible"):
            week2_dev.prepare(self.root, too_many)
        (self.root / "data/processed/gsm8k-heldout.json").unlink()
        with self.assertRaisesRegex(ValueError, "Prepare Week 1 inputs first"):
            week2_dev.prepare(self.root, self.catalog)

    def test_repository_catalog_is_pinned(self):
        catalog = week2_dev.load_catalog(ROOT)
        self.assertEqual(catalog["count_per_task"], 200)
        for spec in catalog["tasks"].values():
            self.assertRegex(spec["source"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(spec["revision"], spec["source"]["url"])


if __name__ == '__main__':
    unittest.main()
