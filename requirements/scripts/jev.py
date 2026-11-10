"""Run the CLI without installing the core package; remote actions are explicit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
