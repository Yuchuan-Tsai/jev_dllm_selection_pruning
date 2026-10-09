"""Run the local CLI without installing packages or accessing the network."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
