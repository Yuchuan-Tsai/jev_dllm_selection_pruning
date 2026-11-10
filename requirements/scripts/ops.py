"""Explicit dataset, GPU, and SSH commands. No operation runs on import."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.data import benchmarks
from jev_dllm.operations import remote, server


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("action", choices=["data-fetch", "data-prepare", "data-check", "storage", "remote-plan", "remote-check", "remote-upload",
        "results-pull", "results-watch", "results-verify", "results-clean", "run-snapshot", "retire-run", "gpu-check", "gpu-check-child",
        "gpu-fetch-model", "gpu-fetch-model-child", "gpu-generate"])
    parser.add_argument("--server-config", type=Path, default=Path("configs/server/local.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/week01/pilot.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/processed/gsm8k-dev.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--verified-digest")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--question-count", type=int, default=1164)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        action = args.action
        if action.startswith("data-"):
            catalog = json.loads((root / "configs/week01/benchmarks.json").read_text())
            result = {"data-fetch": benchmarks.fetch, "data-prepare": benchmarks.prepare, "data-check": benchmarks.audit}[action](root, catalog)
        elif action == "storage":
            result = server.storage_report(root, args.candidates, args.question_count, args.runs)
        elif action in ("run-snapshot", "retire-run"):
            if not args.run_id:
                raise ValueError("An exact run ID is required")
            result = remote.run_snapshot(root, args.run_id) if action == "run-snapshot" else remote.retire_run(root, args.run_id, args.verified_digest or "")
        elif action in ("gpu-check-child", "gpu-fetch-model-child"):
            result = server.gpu_check(root) if action == "gpu-check-child" else server.fetch_model(root, root / args.config)
        elif action.startswith("gpu-"):
            env = server.gpu_environment(root)
            if action == "gpu-generate":
                subprocess.run([sys.executable, str(root / "scripts/ops.py"), "gpu-check-child"], cwd=root, env=env, check=True)
                command = [sys.executable, str(root / "scripts/jev.py"), "generate-pool", "--config", str(root / args.config), "--questions", str(root / args.questions)]
            else:
                command = [sys.executable, str(root / "scripts/ops.py"), action + "-child", "--config", str(root / args.config)]
            subprocess.run(command, cwd=root, env=env, check=True)
            result = {"action": action, "status": "finished"}
        else:
            config = remote.settings(root / args.server_config, root, plan=action == "remote-plan")
            result = remote.operate(root, config, action, args.run_id)
        if result is not None:
            print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped. Partial results are retained; inspect nvidia-smi after stopping a GPU run.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
