import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.backends.device import available_devices, resolve_device


class DeviceTests(unittest.TestCase):
    def test_cpu_does_not_need_torch(self):
        with patch("jev_dllm.backends.device.import_module", side_effect=AssertionError("unexpected import")):
            self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_falls_back_to_cpu_without_torch(self):
        with patch("jev_dllm.backends.device.import_module", side_effect=ImportError):
            self.assertEqual(available_devices(), ("cpu",))
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_mps_selection_and_unavailable_request(self):
        torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                                cuda=SimpleNamespace(is_available=lambda: False))
        with patch("jev_dllm.backends.device.import_module", return_value=torch):
            self.assertEqual(resolve_device("auto"), "mps")
            self.assertEqual(resolve_device("mps"), "mps")
        with patch("jev_dllm.backends.device.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                resolve_device("mps")

    def test_invalid_selection_rejected(self):
        with self.assertRaisesRegex(ValueError, "device must be"):
            resolve_device("tpu")


if __name__ == "__main__":
    unittest.main()
