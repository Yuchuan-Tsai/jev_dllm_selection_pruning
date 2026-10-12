"""Analyze verified local development archives; no transfers, model execution or API calls."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.experiments.backend_comparison import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/processed/gsm8k-dev.json")
    parser.add_argument("--references", type=Path, default=ROOT / "data/private/gsm8k-dev-references.json")
    args = parser.parse_args()
    try:
        print(compare(args.official, args.fast, args.questions, args.references, ROOT / "artifacts/runs"))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
