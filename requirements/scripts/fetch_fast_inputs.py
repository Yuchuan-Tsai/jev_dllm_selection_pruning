"""Fetch only pinned Fast-dLLM source and license; never download model weights."""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.backends.fast_llada import source_files, verify_sources


def main():
    config = json.loads((ROOT / "configs/week01/fast-diversity.json").read_text())
    manifest = source_files(ROOT, config)
    revision = manifest["code_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Fast-dLLM requires an immutable commit")
    for name, expected in manifest["files"].items():
        path = ROOT / "third_party/fast_dllm_v1" / name
        if path.exists():
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"Existing source differs from pin: {name}")
            continue
        with urlopen(f"https://raw.githubusercontent.com/NVlabs/Fast-dLLM/{revision}/{name}", timeout=60) as response:
            content = response.read(expected["bytes"] + 1)
        if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Downloaded source differs from pin: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    verify_sources(ROOT, config)
    print("Fast-dLLM source bundle verified; no model weights downloaded.")


if __name__ == "__main__":
    main()
