"""Stable IDs and trajectory seeds, independent of batching or pruning order."""
import hashlib
import json


def fingerprint(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def trajectory_seed(master_seed: int, question_id: str, candidate_index: int) -> int:
    if candidate_index < 0:
        raise ValueError("candidate_index must be nonnegative")
    return int(fingerprint([master_seed, question_id, candidate_index])[:16], 16) % (2**63)
