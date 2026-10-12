"""Check the existing GPU environment against exact package versions, without mutation."""
import importlib.metadata
from pathlib import Path
import platform
import re
import sys


def parse_lock(path: Path) -> dict[str, str]:
    versions = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise ValueError("Dependency lock must contain exact public package versions only")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in versions:
            raise ValueError("Duplicate package in dependency lock")
        versions[name] = match[2]
    if not versions:
        raise ValueError("Empty dependency lock")
    return versions


def check_lock(path: Path) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("The GPU lock requires Linux x86_64 / Python 3.11")
    expected = parse_lock(path)
    changed = []
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            changed.append({"package": name, "expected": version, "installed": actual})
    if changed:
        raise RuntimeError(f"GPU environment differs from lock: {changed}; do not reinstall a working environment blindly")
    return {"status": "passed", "locked_packages": len(expected), "changed_environment": False,
            "note": "Exact package-version check; not wheel-hash verification or a clean reinstall test."}
