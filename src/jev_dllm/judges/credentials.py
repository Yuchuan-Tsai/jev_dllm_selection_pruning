"""Load only the explicitly requested private API key; never execute shell text."""
import os
from pathlib import Path
import shlex


def load_api_key(path: Path | None = None) -> None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    if path is None:
        raise ValueError("Set TYPESAFE_API_KEY or provide --env-file .env")
    values = []
    for line in path.read_text().splitlines():
        clean = line.strip()
        if clean.startswith("export "):
            clean = clean[7:].lstrip()
        if not clean.startswith("TYPESAFE_API_KEY="):
            continue
        try:
            parts = shlex.split(clean.split("=", 1)[1], comments=True)
        except ValueError:
            raise ValueError("Invalid API key assignment syntax; value not displayed") from None
        if len(parts) != 1 or not parts[0]:
            raise ValueError("API key assignment must have one nonempty value; value not displayed")
        values.append(parts[0])
    if len(values) != 1:
        raise ValueError("Expected exactly one TYPESAFE_API_KEY assignment")
    os.environ["TYPESAFE_API_KEY"] = values[0]
