"""Explicit recovery of a verified interrupted benchmark collection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import collection_recovery as recovery
from jev_dllm.operations import collection_recovery as operations
from jev_dllm.operations import remote


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "audit"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--run-dir", required=True, type=Path)
        if name == "plan":
            cmd.add_argument("--allow-stale-running", action="store_true",
                             help="Confirm no process is active before recovering a manifest left in running state")
    seal = commands.add_parser("seal-stale-copy")
    seal.add_argument("--run-dir", required=True, type=Path)
    seal.add_argument("--confirm-no-process", action="store_true", required=True,
                      help="Confirm pgrep and nvidia-smi showed no active generator on the server")
    for name in ("upload", "source-check", "run", "run-child"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--plan", required=True, type=Path)
        if name == "upload":
            cmd.add_argument("--server-config", type=Path, default=Path("configs/server/local.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "seal-stale-copy":
            result = recovery.seal_stale_copy(root, root / args.run_dir,
                                              confirm_no_process=args.confirm_no_process)
        elif args.action == "plan":
            result = recovery.prepare_plan(root, root / args.run_dir,
                                           allow_stale_running=args.allow_stale_running)
        elif args.action == "audit":
            result = recovery.summary(recovery.inspect_run(root, root / args.run_dir))
        elif args.action == "upload":
            result = operations.upload_plan(root, root / args.plan, remote.settings(root / args.server_config, root))
        elif args.action == "source-check":
            _, _, info = recovery.check_server_source(root, root / args.plan)
            result = recovery.summary(info)
        elif args.action == "run":
            result = operations.launch(root, root / args.plan)
        else:
            result = {"status": "complete", "run_directory": str(recovery.recover(root, root / args.plan))}
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped. Recovery outputs and original source are retained. Check nvidia-smi.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Recovery stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
