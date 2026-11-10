"""Manual MATH500/HumanEval generation; no code execution or paid API requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import collection
from jev_dllm.operations import collection as operations, remote


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = p.add_subparsers(dest="action", required=True)
    for name in ("check", "run", "run-child", "start", "session-run"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--task", choices=tuple(collection.TASKS), required=True)
        if name == "session-run":
            cmd.add_argument("--gpu-id", required=True)
            cmd.add_argument("--hub-cache", required=True)
    cmd = commands.add_parser("upload")
    cmd.add_argument("--server-config", type=Path, default=Path("configs/server/local.json"))
    cmd = commands.add_parser("audit")
    cmd.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "check":
            spec, questions = collection.check(root, args.task)
            result = {"status": "passed", "task": args.task, "questions": len(questions),
                      "candidates": len(questions)*8, "generation_length": 512,
                      "evaluation_ready": False, "gpu_or_api_work_started": False}
        elif args.action == "upload":
            result = operations.upload(root, remote.settings(root / args.server_config, root))
        elif args.action == "start":
            result = operations.start(root, args.task)
        elif args.action == "session-run":
            result = operations.session_run(root, args.task, args.gpu_id, args.hub_cache)
        elif args.action == "run":
            result = operations.launch(root, args.task)
        elif args.action == "run-child":
            result = {"status": "complete", "run_directory": str(collection.generate(root, args.task))}
        else:
            result = collection.audit(root, root / args.run_dir)
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped. Partial records are retained. Inspect nvidia-smi after interruption.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Collection stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
