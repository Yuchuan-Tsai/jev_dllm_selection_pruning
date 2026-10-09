"""Detect duplicate IDs and exact normalized prompt overlap across splits."""
from dataclasses import asdict
import unicodedata
from ..schema import Question
from ..reproducibility import fingerprint


def _prompt_key(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()


def validate_splits(dev: list[Question], heldout: list[Question]) -> dict:
    for expected, rows in (("dev", dev), ("heldout", heldout)):
        if not rows:
            raise ValueError(f"{expected} must not be empty")
        ids, prompts = set(), set()
        for row in rows:
            if row.split != expected or not row.question_id.strip() or not row.prompt.strip():
                raise ValueError(f"Invalid {expected} question: {row.question_id}")
            identity = (row.dataset, row.question_id)
            prompt = _prompt_key(row.prompt)
            if identity in ids or prompt in prompts:
                raise ValueError(f"Duplicate within {expected}: {row.question_id}")
            ids.add(identity)
            prompts.add(prompt)
    shared_ids = {(r.dataset, r.question_id) for r in dev} & {(r.dataset, r.question_id) for r in heldout}
    shared_prompts = {_prompt_key(r.prompt) for r in dev} & {_prompt_key(r.prompt) for r in heldout}
    if shared_ids or shared_prompts:
        raise ValueError("Development and held-out sets overlap")
    return {
        "dev_count": len(dev), "heldout_count": len(heldout),
        "sha256": fingerprint({"dev": [asdict(r) for r in dev], "heldout": [asdict(r) for r in heldout]}),
        "check_scope": "IDs and normalized exact prompts; semantic near-duplicates require review",
    }
