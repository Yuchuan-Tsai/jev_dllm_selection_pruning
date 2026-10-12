"""Opt-in CPU integration checks against pinned upstream source; no model downloads.

Run via `make test-fast-runtime RUNTIME_TEST_PYTHON=/path/to/python` with
torch 2.5.1, transformers 4.38.2 and numpy 1.26.4 installed separately.
The default dependency-free test suite does not discover this directory.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.backends.fast_llada import check_runtime, load_upstream


class FastRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory(prefix="jev-fast-runtime-test-")
        cls.addClassCleanup(cls.cache.cleanup)
        cls.environment = patch.dict("os.environ", {"HF_HOME": cls.cache.name,
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)
        cls.config = json.loads((ROOT / "configs/week01/fast-diversity.json").read_text())
        cls.model_module, _ = load_upstream(ROOT, cls.config)

    def test_prefix_cache_preflight_runs_with_real_tiny_model(self):
        result = check_runtime(ROOT, self.config)
        self.assertEqual(result["status"], "passed")
        self.assertGreaterEqual(result["forward_calls"], 2)
        self.assertLessEqual(result["forward_calls"], 4)
        self.assertFalse(result["model_weights_downloaded"])
        self.assertFalse(result["real_model_validated"])

    def test_legacy_activation_reproduces_the_reported_dimension_error(self):
        config_class = self.model_module.LLaDAConfig

        def legacy_config(**kwargs):
            kwargs.pop("activation_type", None)
            return config_class(**kwargs)

        with patch.object(self.model_module, "LLaDAConfig", side_effect=legacy_config):
            with self.assertRaisesRegex(RuntimeError, r"size of tensor a \(16\).*tensor b \(32\)"):
                check_runtime(ROOT, self.config)


if __name__ == "__main__":
    unittest.main()
