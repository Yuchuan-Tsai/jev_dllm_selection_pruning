"""Explicit, download-free checks of the existing server runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("lock", "fast", "autodl", "autodl-fast"))
    args = parser.parse_args()
    try:
        if args.action.startswith("autodl"):
            import platform
            import numpy
            import torch
            import transformers
            if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
                raise RuntimeError("AutoDL profile requires Linux x86_64 / Python 3.12")
            if not torch.__version__.startswith("2.5.1") or torch.version.cuda != "12.4":
                raise RuntimeError(f"Expected the AutoDL PyTorch 2.5.1 / CUDA 12.4 image; found torch {torch.__version__} / CUDA {torch.version.cuda}")
            if transformers.__version__ != "4.38.2" or numpy.__version__ != "1.26.4":
                raise RuntimeError(f"AutoDL Python packages differ from the provisional profile: transformers={transformers.__version__}, numpy={numpy.__version__}")
            result = {"status": "passed", "profile": "autodl-torch251-py312-cu124-provisional",
                      "python": platform.python_version(), "torch": torch.__version__,
                      "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
                      "numpy": numpy.__version__,
                      "note": "Image compatibility only; full dependency lock and real LLaDA generation still require A800 validation."}
            if args.action == "autodl-fast":
                from jev_dllm.backends.fast_llada import check_runtime
                from jev_dllm.config import load
                config, _ = load(ROOT / "configs/week01/fast-diversity.json")
                result = check_runtime(ROOT, config)
        else:
            from jev_dllm.operations.dependencies import check_lock
            lock = ROOT / "requirements/gpu-linux-py311.lock"
            profile = json.loads((ROOT / "configs/week01/fast-diversity.json").read_text())
            if hashlib.sha256(lock.read_bytes()).hexdigest() != profile["runtime"]["dependency_lock_sha256"]:
                raise ValueError("Dependency lock differs from the profile pin")
            result = check_lock(lock)
        if args.action == "fast":
            from jev_dllm.backends.fast_llada import check_runtime
            from jev_dllm.config import load
            config, _ = load(ROOT / "configs/week01/fast-diversity.json")
            result = check_runtime(ROOT, config)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Runtime check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
