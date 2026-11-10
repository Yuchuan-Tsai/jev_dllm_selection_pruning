"""Explicit Jev diagnostic calls and separate private evaluation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.diagnostics.processbench import evaluate_subset, judge_subset
from jev_dllm.judges.credentials import load_api_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["judge", "evaluate"])
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "judge":
            if args.live:
                load_api_key(args.env_file)
            result = judge_subset(ROOT, ROOT / "artifacts/runs", ROOT / "configs/week01/processbench.json", args.live)
        else:
            if not args.run_dir:
                raise ValueError("Provide --run-dir for a completed diagnostic")
            result = evaluate_subset(ROOT, args.run_dir, ROOT / "artifacts/runs")
        print(result)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
