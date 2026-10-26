"""Opt-in CPU check that trajectory recording leaves the pinned sampler's output unchanged.

Run via `make test-week2-runtime RUNTIME_TEST_PYTHON=/path/to/python` with the
locked GPU dependencies installed (CPU is enough; no model download).
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.backends.fast_llada import load_upstream, run_sampler
from jev_dllm.backends.instrumented import TrajectoryRecorder

MASK = 31
GEN, BLOCK = 8, 2


class RecorderEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = tempfile.TemporaryDirectory(prefix="jev-week2-runtime-")
        cls.addClassCleanup(cache.cleanup)
        environment = patch.dict("os.environ", {"HF_HOME": cache.name, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        environment.start()
        cls.addClassCleanup(environment.stop)
        import torch
        cls.torch = torch
        config = json.loads((ROOT / "configs/week01/fast-diversity.json").read_text())
        model_module, cls.sampler = load_upstream(ROOT, config)
        torch.manual_seed(7)
        tiny = model_module.LLaDAConfig(d_model=16, n_heads=2, n_layers=1, mlp_ratio=2, activation_type="silu",
            vocab_size=32, embedding_size=32, max_sequence_length=32, rope=True, weight_tying=False,
            block_type="llama", flash_attention=False, attention_dropout=0., residual_dropout=0., embedding_dropout=0.)
        cls.model = model_module.LLaDAModelLM(tiny, init_params=True).eval()

        def no_mask_prediction(_model, _inputs, result):
            result.logits[..., MASK] = float("-inf")
            return result
        cls.addClassCleanup(cls.model.register_forward_hook(no_mask_prediction).remove)

    def sample(self, seed):
        torch = self.torch
        torch.manual_seed(seed)
        with torch.inference_mode():
            output, nfe = run_sampler(self.model, self.sampler.generate_with_prefix_cache, torch.tensor([[1, 2, 3]]),
                steps=GEN, gen_length=GEN, block_length=BLOCK, temperature=1., threshold=.9, mask_id=MASK)
        return output, nfe, torch.get_rng_state()

    def test_recording_does_not_change_tokens_nfe_or_random_state(self):
        recorder = TrajectoryRecorder(self.sampler.generate_with_prefix_cache, self.model, self.torch,
                                      (1, 2, 3), GEN // BLOCK, mask_id=MASK)
        for seed in (11, 12, 13):
            with self.subTest(seed=seed):
                plain = self.sample(seed)
                recorder.install()
                try:
                    recorder.start(3, 3 + GEN)
                    recorded = self.sample(seed)
                    trace = recorder.finish(GEN)
                finally:
                    recorder.remove()
                self.assertTrue(self.torch.equal(plain[0], recorded[0]))
                self.assertEqual(plain[1], recorded[1])
                self.assertTrue(self.torch.equal(plain[2], recorded[2]))
                self.assert_trace(trace, recorded[0][0, 3:].tolist(), recorded[1])
        self.assertIs(self.sampler.get_transfer_index, recorder.original)
        self.assertIs(recorder.globals, vars(self.sampler))  # The real module namespace was patched.

    def assert_trace(self, trace, tokens, nfe):
        self.assertTrue(all(0 <= p <= 1 for p in trace["commit_probability"]))
        self.assertTrue(all(1 <= s <= nfe for s in trace["commit_forward"]))
        self.assertEqual(trace["confidence"]["scored_answer_tokens"], GEN)
        self.assertAlmostEqual(trace["confidence"]["mean_probability"], sum(trace["commit_probability"]) / GEN)
        self.assertEqual([c["completed_blocks"] for c in trace["checkpoints"]], [1, 2, 3])
        previous = 0
        for checkpoint in trace["checkpoints"]:
            done = checkpoint["completed_blocks"] * BLOCK
            state, draft = checkpoint["state_ids"], checkpoint["argmax_fill_ids"]
            self.assertEqual(state[:done], tokens[:done])  # Completed blocks are final.
            self.assertTrue(all(t == MASK for t in state[done:]))
            self.assertNotIn(MASK, draft)
            self.assertEqual(draft[:done], tokens[:done])
            self.assertGreater(checkpoint["nfe_before"], previous)
            previous = checkpoint["nfe_before"]
            self.assertIsNone(checkpoint["gpu_seconds_before"])  # CPU run: no CUDA events.

    def test_double_install_is_refused(self):
        recorder = TrajectoryRecorder(self.sampler.generate_with_prefix_cache, self.model, self.torch,
                                      (1,), GEN // BLOCK, mask_id=MASK)
        recorder.install()
        try:
            with self.assertRaises(RuntimeError):
                recorder.install()
        finally:
            recorder.remove()


if __name__ == "__main__":
    unittest.main()
