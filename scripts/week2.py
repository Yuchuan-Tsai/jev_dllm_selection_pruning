"""Week 2 development data and N=16 pool generation; no paid API requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.data import week2_dev
from jev_dllm.experiments import week2_pool
from jev_dllm.operations import remote, week2 as operations


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = p.add_subparsers(dest="action", required=True)
    for name in ("data-fetch", "data-prepare", "data-check"):
        commands.add_parser(name)
    commands.add_parser("upload").add_argument("--server-config", type=Path, default=Path("configs/server/local.json"))
    for name in ("pool-check", "pool-run", "pool-run-child", "pool-start", "pool-session-run"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--task", choices=week2_pool.TASKS, required=True)
        if name != "pool-check":
            cmd.add_argument("--resume-from", type=Path)
        if name == "pool-session-run":
            cmd.add_argument("--gpu-id", required=True)
            cmd.add_argument("--hub-cache", required=True)
    for name in ("equivalence", "equivalence-child"):
        commands.add_parser(name).add_argument("--reference", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    resume = getattr(args, "resume_from", None)
    resume = (root / resume).resolve() if resume else None
    try:
        if args.action.startswith("data-"):
            catalog = week2_dev.load_catalog(root)
            if args.action == "data-fetch":
                week2_dev.fetch(root, catalog)
                result = {"status": "verified"}
            else:
                result = (week2_dev.prepare if args.action == "data-prepare" else week2_dev.audit)(root, catalog)
        elif args.action == "upload":
            result = operations.upload(root, remote.settings(root / args.server_config, root))
        elif args.action == "pool-check":
            profile, base, questions = week2_pool.load_profile(root, args.task)
            result = {"status": "passed", "task": args.task, "questions": len(questions),
                      "candidates": len(questions) * profile["candidates_per_question"],
                      "generation_length": base["generator"]["generation_length"],
                      "checkpoint_completed_blocks": profile["checkpoint_completed_blocks"], "gpu_or_api_work_started": False}
        elif args.action == "pool-run":
            result = operations.child(root, "pool-run-child", "--task", args.task,
                                      *(["--resume-from", str(resume)] if resume else []))
        elif args.action == "pool-run-child":
            result = {"status": "complete", "run_directory": str(week2_pool.generate(root, args.task, resume))}
        elif args.action == "pool-start":
            result = operations.start(root, args.task, resume)
        elif args.action == "pool-session-run":
            result = operations.session_run(root, args.task, args.gpu_id, args.hub_cache, resume)
        elif args.action == "equivalence":
            result = operations.child(root, "equivalence-child", "--reference", str((root / args.reference).resolve()))
        else:
            run, report = week2_pool.equivalence(root, (root / args.reference).resolve())
            result = {"run_directory": str(run), **{k: report[k] for k in ("status", "compared", "identical")}}
            if report["mismatches"]:
                result["first_mismatches"] = report["mismatches"][:5]
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") != "different" else 1
    except KeyboardInterrupt:
        print("Stopped. Partial records are retained; resume with --resume-from RUN_DIR.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Week 2 command stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
