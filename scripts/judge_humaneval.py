"""HumanEval Jev: offline prepare, explicit live selection, then private evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import humaneval_jev as workflow
from jev_dllm.judges.credentials import load_api_key


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "live", "evaluate"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--score-run", type=Path)
    parser.add_argument("--approved-calls", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = lambda p: (root / p).resolve() if p else None
    try:
        if args.action == "prepare":
            if not args.run_dir:
                parser.error("prepare requires --run-dir")
            out = workflow.prepare(root, path(args.run_dir))
        elif args.action == "live":
            if not args.plan_dir or args.approved_calls is None:
                parser.error("live requires --plan-dir and --approved-calls")
            # Validate before reading credentials, and before any network request.
            workflow.inspect(root, path(args.plan_dir))
            load_api_key(path(args.env_file))
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise ValueError("Set TYPESAFE_API_KEY in the ignored .env file")
            out = workflow.execute(root, path(args.plan_dir), args.approved_calls)
        else:
            if not args.plan_dir or not args.score_run:
                parser.error("evaluate requires --plan-dir and --score-run")
            out = workflow.evaluate(root, path(args.plan_dir), path(args.score_run))
        print(json.dumps({"status": "complete", "action": args.action, "output": str(out)}, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped; saved evidence retained. Do not restart a paid batch blindly.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"HumanEval Jev stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
