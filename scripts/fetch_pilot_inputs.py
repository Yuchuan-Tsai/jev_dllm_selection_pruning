"""Explicit public source/data download; never downloads weights or calls Jev."""
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/week01/pilot.json").read_text())
    manifest = json.loads((root / config["datasets"]["manifest_path"]).read_text())
    data_revision = config["datasets"]["revision"]
    sampler = config["backends"]["official_llada"]
    if not all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in (data_revision, sampler["code_commit"])):
        raise ValueError("Downloads require immutable source commits")
    urls = {
        "third_party/llada/generate.py": (
            f"https://raw.githubusercontent.com/ML-GSAI/LLaDA/{sampler['code_commit']}/generate.py",
            sampler["source_sha256"]),
        **{f"data/raw/gsm8k-{split}.jsonl": (
            f"https://raw.githubusercontent.com/openai/grade-school-math/{data_revision}/grade_school_math/data/{split}.jsonl",
            manifest["sources"][split]["sha256"]) for split in ("train", "test")},
    }
    for name, (url, expected) in urls.items():
        path = root / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing file differs from pin: {name}")
            print(f"Verified cached file: {name}")
            continue
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Downloaded content differs from pin: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        print(f"Downloaded and verified: {name}")


if __name__ == "__main__":
    main()
