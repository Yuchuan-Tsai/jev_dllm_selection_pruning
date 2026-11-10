"""Week 4 Decision Point A: check the frozen final selector; offline, no labels or paid API requests."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import frozen
from jev_dllm.selection import frozen_selector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_subparsers(dest="action", required=True).add_parser("selector-check")
    root = p.parse_args().root.resolve()
    try:
        record = frozen_selector.check(root)
        result = {"status": "passed", "record": frozen_selector.RECORD,
                  "record_sha256": frozen.sha256(root / frozen_selector.RECORD),
                  "selector": record["selector"]["name"], "jev_model": record["jev"]["model"],
                  "protocol": record["jev"]["protocol"], "pinned_files": len(record["files_sha256"]),
                  "evidence": frozen_selector.verify_evidence(root, record), "paid_calls_sent": 0}
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Week 4 selector check stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
