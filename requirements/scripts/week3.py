"""Week 3 selector comparison: open-model judge runs (server GPU) and local analysis; no paid API calls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.evaluation import week3
from jev_dllm.judges import open_models
from jev_dllm.operations import server


def latest(base: Path, prefix: str, complete=True) -> Path:
    """Newest run directory named prefix-*, optionally only one whose manifest says complete."""
    runs = sorted(base.glob(prefix + "-*"))
    if complete:
        runs = [r for r in runs if (r / "manifest.json").exists()
                and json.loads((r / "manifest.json").read_text()).get("status") == "complete"]
    if not runs:
        raise ValueError(f"No {'complete ' if complete else ''}{prefix} run under {base}")
    return runs[-1]


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = p.add_subparsers(dest="action", required=True)
    commands.add_parser("judges-check")
    commands.add_parser("judges-fetch").add_argument("--kind", choices=open_models.KINDS, required=True)
    for name in ("judge-run", "judge-run-child"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--kind", choices=open_models.KINDS, required=True)
        cmd.add_argument("--task", choices=("gsm8k", "math", "mbpp"), required=True)
        cmd.add_argument("--pool-dir", type=Path, help="Server pool run; default: newest complete w2pool-<task>")
        cmd.add_argument("--limit", type=int, help="Smoke test: score only the first K questions")
    cmd = commands.add_parser("analyze")
    cmd.add_argument("--task", choices=("gsm8k", "math", "mbpp"), required=True)
    cmd.add_argument("--labels-dir", type=Path, help="Default: newest complete w2labels-<task>")
    cmd.add_argument("--math-python", type=Path, default=Path(".venv-math/bin/python"))
    cmd.add_argument("--jev-plan", type=Path, help="Completed Week 2 Jev plan directory (with live/)")
    cmd.add_argument("--judge-run", type=Path, help="Verified local copy; default: newest w3judge-<task> if any")
    cmd.add_argument("--prm-run", type=Path, help="Verified local copy; default: newest w3prm-<task> if any")
    args = p.parse_args()
    root = args.root.resolve()
    runs = root / "artifacts/runs"
    try:
        if args.action == "judges-check":
            result = open_models.check_environment(root)
        elif args.action == "judges-fetch":
            cache = os.environ.get("JEV_SHARED_HUB_CACHE")
            if not cache:
                raise ValueError("Export JEV_SHARED_HUB_CACHE to the model cache before downloading")
            os.environ["HF_HUB_CACHE"] = cache
            result = open_models.fetch(root, args.kind)
        elif args.action == "judge-run":
            # One GPU, the model cache and offline Hub settings, as for generation.
            env = server.gpu_environment(root)
            child = [sys.executable, str(root / "scripts/week3.py"), "--root", str(root), "judge-run-child",
                     "--kind", args.kind, "--task", args.task]
            child += ["--pool-dir", str(args.pool_dir)] if args.pool_dir else []
            child += ["--limit", str(args.limit)] if args.limit else []
            subprocess.run(child, env=dict(env, HF_HUB_OFFLINE="1"), check=True)
            result = {"status": "finished", "kind": args.kind, "task": args.task}
        elif args.action == "judge-run-child":
            pool = (root / args.pool_dir).resolve() if args.pool_dir else latest(runs, f"w2pool-{args.task}")
            result = {"status": "complete", "run_directory": str(open_models.run(root, args.kind, pool, args.limit))}
        else:
            lab = root / "artifacts/downloads/lab"
            labels = (root / args.labels_dir).resolve() if args.labels_dir else latest(runs, f"w2labels-{args.task}")
            def copy(given, prefix):
                if given:
                    return (root / given).resolve()
                found = sorted(lab.glob(prefix + "-*"))
                return found[-1] if found else None
            out = week3.analyze(root, labels, (root / args.math_python) if args.task == "math" else None,
                                (root / args.jev_plan).resolve() if args.jev_plan else None,
                                copy(args.judge_run, f"w3judge-{args.task}"),
                                copy(args.prm_run, f"w3prm-{args.task}") if args.task != "mbpp" else None)
            print((out / "summary.md").read_text())
            result = {"status": "complete", "analysis_directory": str(out)}
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped. Partial outputs are retained; rerun the command for a fresh directory.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Week 3 command stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
