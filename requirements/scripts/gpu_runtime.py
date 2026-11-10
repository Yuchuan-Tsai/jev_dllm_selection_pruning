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
    parser.add_argument("action", choices=("lock", "fast"))
    args = parser.parse_args()
    try:
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
