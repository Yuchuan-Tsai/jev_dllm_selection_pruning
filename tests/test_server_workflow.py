import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.data.benchmarks import partition, verify_raw, write_unchanged, select_diagnostic
from jev_dllm.operations.remote import run_snapshot, verify_copy, retire_run, snapshot_digest, settings, pull, operate
from jev_dllm.operations.server import gpu_environment, storage_report
from jev_dllm.operations.server import gpu_check
from jev_dllm.experiments.selection import evaluate_pools


class BenchmarkSeparationTests(unittest.TestCase):
    def test_math_public_has_no_solution_or_answer(self):
        public, private = partition("math500", [{"unique_id": "test/algebra/1", "problem": "Find x", "answer": "2", "solution": "private reasoning", "extra": "private"}])
        self.assertEqual(set(public[0]), {"question_id", "dataset", "split", "prompt"})
        self.assertEqual(public[0]["split"], "heldout")
        self.assertEqual(private[0]["answer"], "2")

    def test_humaneval_hidden_tests_are_private(self):
        public, private = partition("humaneval", [{"task_id": "HumanEval/0", "prompt": "def f():", "test": "secret tests", "canonical_solution": "return 1", "entry_point": "f"}])
        self.assertNotIn("secret tests", json.dumps(public))
        self.assertNotIn("return 1", json.dumps(public))
        self.assertEqual(private[0]["hidden_tests"], "secret tests")

    def test_mbpp_development_uses_only_train_and_validation_ranges(self):
        rows = [
            {"task_id": i, "text": f"task {i}", "code": "secret code", "test_list": ["secret test"]}
            for i in (1, 11, 510, 511, 600, 601, 974)
        ]
        public, private = partition("mbpp", rows)
        self.assertEqual([q["question_id"] for q in public], ["mbpp/511", "mbpp/600", "mbpp/601", "mbpp/974"])
        self.assertTrue(all(q["split"] == "dev" for q in public))
        self.assertNotIn("secret code", json.dumps(public))
        self.assertNotIn("secret test", json.dumps(public))
        self.assertEqual(len(private), len(public))

    def test_duplicate_ids_rejected(self):
        row = {"unique_id": "x", "problem": "p", "answer": "1"}
        with self.assertRaises(ValueError):
            partition("math500", [row, row])

    def test_diagnostic_excludes_generation_questions_without_using_labels(self):
        rows = [{"id": str(i), "problem": "  Heldout question " if i == 0 else f"other {i}", "label": i} for i in range(6)]
        selected, excluded = select_diagnostic(rows, {"heldout question"}, 3, 1, "revision")
        self.assertEqual(excluded, 1)
        self.assertNotIn("0", [r["id"] for r in selected])
        for row in rows:
            row["label"] = -1
        repeat, _ = select_diagnostic(rows, {"heldout question"}, 3, 1, "revision")
        self.assertEqual([r["id"] for r in selected], [r["id"] for r in repeat])

    def test_diagnostic_does_not_silently_shrink_coverage(self):
        with self.assertRaises(ValueError):
            select_diagnostic([{"id": "x", "problem": "protected"}], {"protected"}, 1, 1, "revision")

    def test_changed_data_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            write_unchanged(path, b"original")
            with self.assertRaises(ValueError):
                write_unchanged(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")

    def test_bad_source_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"; path.write_bytes(b"123")
            with self.assertRaises(ValueError):
                verify_raw(path, {"bytes": 3, "sha256": "0" * 64})

    def test_math500_cannot_use_the_numeric_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.json"
            questions.write_text(json.dumps([{"question_id": "q", "dataset": "math500", "split": "dev", "prompt": "Find x"}]))
            candidates = root / "candidates.jsonl"
            candidates.write_text(json.dumps({"question_id": "q", "candidate_id": "c", "seed": 1, "text": "Final answer: 2", "completed": True, "truncated": False}) + "\n")
            with self.assertRaisesRegex(ValueError, "GSM8K only"):
                evaluate_pools(questions, candidates, root / "unopened-private-reference.json", root / "runs")


class TransferSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "server"
        self.run_id = "llada-dev-20261009-example"
        self.run = self.root / "artifacts/runs" / self.run_id
        self.run.mkdir(parents=True)
        (self.run / "manifest.json").write_text(json.dumps({"status": "complete"}))
        (self.run / "candidates.jsonl").write_text('sample\n')
        self.local = Path(self.tmp.name) / "local"
        shutil.copytree(self.run, self.local / self.run_id)

    def tearDown(self):
        self.tmp.cleanup()

    def test_verified_terminal_run_can_be_retired(self):
        snapshot = run_snapshot(self.root, self.run_id)
        digest = verify_copy(self.local, snapshot)
        retire_run(self.root, self.run_id, digest)
        self.assertFalse(self.run.exists())
        self.assertTrue((self.local / self.run_id / "candidates.jsonl").exists())

    def test_corrupt_local_copy_blocks_retirement(self):
        snapshot = run_snapshot(self.root, self.run_id)
        (self.local / self.run_id / "candidates.jsonl").write_text("broken")
        with self.assertRaises(ValueError):
            verify_copy(self.local, snapshot)
        self.assertTrue(self.run.exists())

    def test_local_extra_file_rejected(self):
        (self.local / self.run_id / "extra.txt").write_text("do not mix analysis into immutable archive")
        with self.assertRaises(ValueError):
            verify_copy(self.local, run_snapshot(self.root, self.run_id))

    def test_changed_remote_copy_blocks_retirement(self):
        digest = snapshot_digest(run_snapshot(self.root, self.run_id))
        (self.run / "candidates.jsonl").write_text("changed")
        with self.assertRaises(ValueError):
            retire_run(self.root, self.run_id, digest)
        self.assertTrue(self.run.exists())

    def test_active_run_cannot_be_retired(self):
        (self.run / "manifest.json").write_text(json.dumps({"status": "running"}))
        with self.assertRaises(ValueError):
            run_snapshot(self.root, self.run_id)

    def test_symlink_in_run_rejected(self):
        (self.run / "link").symlink_to(self.local / self.run_id / "candidates.jsonl")
        with self.assertRaises(ValueError):
            run_snapshot(self.root, self.run_id)

    def test_run_path_traversal_rejected(self):
        for name in ("../shared", "/shared", ".", "a/b"):
            with self.assertRaises(ValueError):
                run_snapshot(self.root, name)

    def test_remote_config_rejects_shared_cleanup_root(self):
        config = Path(self.tmp.name) / "config.json"
        config.write_text(json.dumps({"ssh_host": "lab", "remote_root": "/shared/rsaas/huggingface", "local_results": "archive"}))
        with self.assertRaises(ValueError):
            settings(config, self.root)

    def test_remote_config_rejects_unfilled_connection(self):
        config = Path(self.tmp.name) / "config.json"
        config.write_text(json.dumps({"ssh_host": "YOUR_SSH_ALIAS", "remote_root": "/home/YOUR_USERNAME/project", "local_results": "archive"}))
        with self.assertRaises(ValueError):
            settings(config, self.root)
        self.assertEqual(settings(config, self.root, plan=True)["ssh_host"], "YOUR_SSH_ALIAS")

    def test_pull_does_not_delete_or_write_in_place(self):
        config = {"ssh_host": "lab", "remote_root": "/home/person/project", "local_results": str(self.local)}
        with patch("jev_dllm.operations.remote.subprocess.run") as command:
            pull(config)
        args = command.call_args.args[0]
        self.assertNotIn("--delete", args)
        self.assertNotIn("--remove-source-files", args)
        self.assertNotIn("--inplace", args)

    def test_clean_never_sends_delete_when_local_verification_fails(self):
        with patch("jev_dllm.operations.remote.verify_remote", side_effect=ValueError("bad copy")), patch("jev_dllm.operations.remote.subprocess.run") as command:
            with self.assertRaises(ValueError):
                operate(self.root, {}, "results-clean", self.run_id)
            command.assert_not_called()


class GPUSetupTests(unittest.TestCase):
    def test_v100_native_precision_is_rejected(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(major=7), is_bf16_supported=lambda: True))
        with patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaisesRegex(RuntimeError, "Native BF16"):
                gpu_check(Path("/tmp"))

    def test_multiple_visible_gpus_are_rejected(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
        with patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                gpu_check(Path("/tmp"))

    def test_requires_one_explicit_gpu(self):
        for device in ("", "0,1", "-1", "all"):
            with patch.dict(os.environ, {"GPU_ID": device}):
                with self.assertRaises(ValueError):
                    gpu_environment(Path("/tmp/project"))

    def test_credentials_stay_outside_shared_cache(self):
        with patch.dict(os.environ, {"GPU_ID": "2", "HF_HOME": "/shared/rsaas/huggingface"}):
            env = gpu_environment(Path("/tmp/project"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertNotIn("/shared/", env["HF_HOME"])
        self.assertNotIn("/shared/", env["HF_TOKEN_PATH"])
        self.assertEqual(env["HF_HUB_CACHE"], "/shared/rsaas/huggingface/hub")

    def test_storage_scales_with_candidates_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = storage_report(Path(tmp), candidates=8, runs=1)
            large = storage_report(Path(tmp), candidates=16, runs=2)
        self.assertEqual(large["scenario"]["estimated_output_bytes_all_runs"], 4 * small["scenario"]["estimated_output_bytes_all_runs"])


class InterruptedGenerationTests(unittest.TestCase):
    def test_interruption_leaves_a_terminal_manifest(self):
        from jev_dllm.experiments.generation import generate_pool
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"; prompt.write_text("Question: {question}")
            questions = root / "questions.json"
            questions.write_text(json.dumps([{"question_id": "q", "dataset": "gsm8k", "split": "dev", "prompt": "2+3?"}]))
            config = {"generator": {"candidates_per_question": 4, "selected_backend": "official_llada"}, "runtime": {"master_seed": 1},
                "prompts": {"generation_math": {"path": "prompt.txt", "sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()}}}
            backend = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: "fixture GPU")))
            def interrupt(*args, **kwargs):
                raise KeyboardInterrupt()
            backend.generate = interrupt
            with patch("jev_dllm.experiments.generation.OfficialLLaDA", return_value=backend), patch("jev_dllm.experiments.generation.importlib.metadata.version", return_value="fixture"), patch("jev_dllm.experiments.generation.subprocess.run", return_value=SimpleNamespace(stdout="", returncode=1)):
                with self.assertRaises(KeyboardInterrupt):
                    generate_pool(root, config, questions, root / "artifacts/runs")
            run = next((root / "artifacts/runs").iterdir())
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "interrupted")
            self.assertEqual(run_snapshot(root, run.name)["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
