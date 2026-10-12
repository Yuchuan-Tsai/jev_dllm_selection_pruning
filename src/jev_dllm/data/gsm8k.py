"""Prepare a small reproducible GSM8K pilot from local upstream JSONL files."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from ..evaluation.answers import extract_final_answer, numeric_value
from ..reproducibility import fingerprint
from ..schema import Question
from .splits import validate_splits


def prepare(root: Path, train: Path, test: Path, revision: str,
            dev_count: int = 20, heldout_count: int = 100, seed: int = 20261008) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be an immutable Git commit")
    public, private, source = {}, {}, {}
    for split, upstream, path, count in (("dev", "train", train, dev_count),
                                         ("heldout", "test", test, heldout_count)):
        content = path.read_bytes()
        rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
        if type(count) is not int or not 0 < count <= len(rows):
            raise ValueError(f"Invalid requested {split} count")
        # Selection depends on source row IDs, never answers or model outcomes.
        indices = sorted(range(len(rows)), key=lambda i: fingerprint([seed, revision, upstream, i]))[:count]
        questions, references = [], []
        for i in indices:
            row = rows[i]
            qid = f"gsm8k/{upstream}/{i:05}"
            prompt = row["question"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Invalid prompt: {qid}")
            answer = numeric_value(extract_final_answer(row["answer"]))
            if answer is None:
                raise ValueError(f"Invalid numeric reference: {qid}")
            questions.append(Question(qid, "gsm8k", split, prompt))
            references.append({"question_id": qid, "answer": str(answer)})
        public[split], private[split] = questions, references
        source[upstream] = {"sha256": hashlib.sha256(content).hexdigest(),
                            "source_rows": len(rows), "selected_row_indices": indices}
    checks = validate_splits(public["dev"], public["heldout"])
    files = {}
    for split in public:
        files[f"data/processed/gsm8k-{split}.json"] = [asdict(q) for q in public[split]]
        files[f"data/private/gsm8k-{split}-references.json"] = private[split]
    encoded = {name: (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
               for name, value in files.items()}
    manifest = {
        "dataset": "gsm8k", "repository": "https://github.com/openai/grade-school-math",
        "revision": revision, "seed": seed, "selection": "lowest_sha256_of_seed_revision_split_row_index",
        "sources": source, "split_checks": checks,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in encoded.items()},
        "question_ids": {s: [q.question_id for q in rows] for s, rows in public.items()},
        "limitation": "Exact/normalized overlap checks do not establish semantic deduplication.",
    }
    manifest_path = root / "data/manifests/gsm8k-week01.json"
    encoded["data/manifests/gsm8k-week01.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, raw in encoded.items():
        path = root / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite different prepared data: {name}")
    for name, raw in encoded.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return manifest_path
