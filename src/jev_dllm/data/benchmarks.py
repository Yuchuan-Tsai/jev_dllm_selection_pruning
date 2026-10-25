"""Pinned small benchmark inputs; references never enter public records."""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from .gsm8k import prepare as prepare_gsm8k
from .splits import _prompt_key
from ..reproducibility import fingerprint
from ..schema import Question


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def write_unchanged(path: Path, raw: bytes):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite different prepared input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def verify_raw(path: Path, source: dict):
    raw = path.read_bytes()
    if (("bytes" in source and len(raw) != source["bytes"])
            or ("sha256" in source and hashlib.sha256(raw).hexdigest() != source["sha256"])):
        raise ValueError(f"Input differs from recorded source: {path.name}")
    return raw


def fetch(root: Path, catalog: dict):
    for spec in catalog["benchmarks"].values():
        for source in spec["sources"]:
            path = root / source["path"]
            if not path.exists():
                with urlopen(source["url"], timeout=60) as response:
                    # All sources are small text datasets; fail on an unexpected large response.
                    raw = response.read(source.get("bytes", source.get("max_bytes", 2_000_000)) + 1)
                if "bytes" in source and len(raw) != source["bytes"]:
                    raise ValueError(f"Downloaded input has an unexpected size: {path.name}")
                if len(raw) > source.get("max_bytes", len(raw)):
                    raise ValueError(f"Downloaded input exceeds the configured size limit: {path.name}")
                if "sha256" in source and hashlib.sha256(raw).hexdigest() != source["sha256"]:
                    raise ValueError(f"Downloaded input failed verification: {path.name}")
                write_unchanged(path, raw)
            verify_raw(path, source)
            print(f"Verified {source['path']} ({(root / source['path']).stat().st_size} bytes)")


def source_rows(root: Path, source: dict):
    raw = verify_raw(root / source["path"], source)
    if source["path"].endswith(".gz"):
        raw = gzip.decompress(raw)
    rows = json.loads(raw) if source["format"] == "json" else [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != source["rows"]:
        raise ValueError("Unexpected source row count")
    return rows


def partition(name: str, rows: list[dict]):
    public, private = [], []
    for row in rows:
        if name == "math500":
            qid = "math500/" + row["unique_id"]
            question = Question(qid, name, "heldout", row["problem"])
            reference = {"question_id": qid, "answer": row["answer"]}
        elif name == "humaneval":
            qid = row["task_id"]
            question = Question(qid, name, "heldout", row["prompt"])
            reference = {"question_id": qid, "entry_point": row["entry_point"],
                         "hidden_tests": row["test"], "canonical_solution": row["canonical_solution"]}
        elif name == "mbpp":
            task_id = int(row["task_id"])
            if not 511 <= task_id <= 974:
                continue
            qid = f"mbpp/{task_id}"
            question = Question(qid, name, "dev", row["text"])
            reference = {"question_id": qid, "reference_code": row["code"],
                         "test_setup_code": row.get("test_setup_code", ""),
                         "tests": row.get("test_list", []),
                         "challenge_tests": row.get("challenge_test_list", [])}
        else:
            raise ValueError("Unknown generation benchmark")
        if not question.prompt.strip():
            raise ValueError("Empty question")
        public.append(asdict(question))
        private.append(reference)
    if len({q["question_id"] for q in public}) != len(public):
        raise ValueError("Duplicate source IDs")
    return public, private


def save_outputs(root: Path, name: str, outputs: dict, metadata: dict):
    raw = {path: encoded(value) for path, value in outputs.items()}
    manifest = {"dataset": name, **metadata,
                "files": {path: hashlib.sha256(value).hexdigest() for path, value in raw.items()}}
    raw[f"data/manifests/{name}-week01.json"] = encoded(manifest)
    # Check every destination before writing any file for this benchmark.
    for path, value in raw.items():
        if (root / path).exists() and (root / path).read_bytes() != value:
            raise ValueError(f"Prepared input changed: {path}")
    for path, value in raw.items():
        write_unchanged(root / path, value)


def select_diagnostic(rows, protected_prompts, count, seed, revision):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate ProcessBench source IDs")
    eligible = [r for r in rows if _prompt_key(r["problem"]) not in protected_prompts]
    if len(eligible) < count:
        raise ValueError("Not enough nonoverlapping diagnostic items; revise coverage explicitly")
    ordered = sorted(eligible, key=lambda r: fingerprint([seed, revision, r["id"]]))
    return ordered[:count], len(rows) - len(eligible)


def prepare(root: Path, catalog: dict):
    specs = catalog["benchmarks"]
    gsm = specs["gsm8k"]
    for source in gsm["sources"]:
        verify_raw(root / source["path"], source)
    prepare_gsm8k(root, root / gsm["sources"][0]["path"], root / gsm["sources"][1]["path"],
                  gsm["revision"], dev_count=20, heldout_count=500)
    for name in ("math500", "humaneval"):
        spec = specs[name]
        public, private = partition(name, source_rows(root, spec["sources"][0]))
        save_outputs(root, name, {
            f"data/processed/{name}-heldout.json": public,
            f"data/private/{name}-heldout-references.json": private,
        }, {"revision": spec["revision"], "sources": spec["sources"],
            "heldout_count": len(public), "dev_count": 0,
            "question_ids": [q["question_id"] for q in public],
            "development_status": "Separate development source still required; do not tune on this heldout set"})
    mbpp = specs["mbpp"]
    mbpp_rows = source_rows(root, mbpp["sources"][0])
    mbpp_public, mbpp_private = partition("mbpp", mbpp_rows)
    mbpp_train = [q for q in mbpp_public if 601 <= int(q["question_id"].split("/")[1]) <= 974]
    mbpp_validation = [q for q in mbpp_public if 511 <= int(q["question_id"].split("/")[1]) <= 600]
    if len(mbpp_public) != 464 or len(mbpp_train) != 374 or len(mbpp_validation) != 90:
        raise ValueError("MBPP official development ID ranges did not produce 374 train and 90 validation tasks")
    source = mbpp["sources"][0]
    mbpp_source_record = {**source, "bytes": (root / source["path"]).stat().st_size,
                          "sha256": hashlib.sha256((root / source["path"]).read_bytes()).hexdigest()}
    humaneval_questions = json.loads((root / "data/processed/humaneval-heldout.json").read_text())
    humaneval_prompts = {_prompt_key(q["prompt"]) for q in humaneval_questions}
    mbpp_exact_humaneval_overlap = [q["question_id"] for q in mbpp_public
                                    if _prompt_key(q["prompt"]) in humaneval_prompts]
    save_outputs(root, "mbpp", {
        "data/processed/mbpp-dev.json": mbpp_public,
        "data/private/mbpp-dev-references.json": mbpp_private,
    }, {"revision": mbpp["revision"], "sources": [mbpp_source_record],
        "dev_count": len(mbpp_public), "train_count": len(mbpp_train),
        "validation_count": len(mbpp_validation), "question_ids": [q["question_id"] for q in mbpp_public],
        "split_policy": "official task IDs 511-600 validation and 601-974 train; excludes 1-10 few-shot and 11-510 test",
        "exact_prompt_overlap_with_humaneval": mbpp_exact_humaneval_overlap,
        "semantic_near_duplicate_review": "pending; normalized exact overlap is not a semantic review",
        "heldout_humaneval_used": False,
        "execution_status": "Reference tests are private; candidate code execution requires a future isolated worker"})
    # Record exact overlaps without exposing labels or treating diagnostics as selector data.
    protected = set()
    for name in ("gsm8k", "math500"):
        protected.update(_prompt_key(q["prompt"]) for q in json.loads((root / f"data/processed/{name}-heldout.json").read_text()))
    protected.update(_prompt_key(q["prompt"]) for q in json.loads((root / "data/processed/gsm8k-dev.json").read_text()))
    pb = specs["processbench"]
    outputs, subsets = {}, {}
    for source in pb["sources"]:
        subset = source["subset"]
        rows = source_rows(root, source)
        chosen, excluded = select_diagnostic(rows, protected, pb["diagnostic_count_per_subset"], catalog["seed"], pb["revision"])
        public = [{"question_id": f"processbench/{r['id']}", "subset": subset,
                   "problem": r["problem"], "steps": r["steps"]} for r in chosen]
        private = [{"question_id": f"processbench/{r['id']}", "label": r["label"],
                    "final_answer_correct": r["final_answer_correct"]} for r in chosen]
        outputs[f"data/processed/processbench-{subset}-diagnostic.json"] = public
        outputs[f"data/private/processbench-{subset}-labels.json"] = private
        subsets[subset] = {"source_count": len(rows), "diagnostic_count": len(chosen),
                           "question_ids": [q["question_id"] for q in public],
                           "excluded_source_rows_matching_generation_prompts": excluded,
                           "exact_prompt_overlap_with_generation_inputs": sum(_prompt_key(r["problem"]) in protected for r in chosen)}
    save_outputs(root, "processbench", outputs, {
        "revision": pb["revision"], "official_code_revision": pb["official_code_revision"],
        "sources": pb["sources"], "seed": catalog["seed"], "subsets": subsets,
        "selection": "exclude_normalized_generation_prompts_then_lowest_sha256_of_seed_revision_id",
        "restriction": "Separate diagnostic only. Exact generation-prompt overlaps are excluded; semantic overlap review remains necessary. Keep diagnostic outcomes out of selector tuning until that review.",
    })
    return audit(root, catalog)


def audit(root: Path, catalog: dict):
    result = {}
    for name, spec in catalog["benchmarks"].items():
        for source in spec["sources"]:
            verify_raw(root / source["path"], source)
        manifest = json.loads((root / f"data/manifests/{name}-week01.json").read_text())
        manifest_sources = manifest.get("sources", [])
        for source in manifest_sources if isinstance(manifest_sources, list) else []:
            if source.get("path") in {s["path"] for s in spec["sources"]} and "sha256" in source:
                raw = (root / source["path"]).read_bytes()
                if len(raw) != source["bytes"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
                    raise ValueError(f"Raw source differs from prepared manifest: {source['path']}")
        for relative, digest in manifest["files"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Prepared file differs from manifest: {relative}")
        result[name] = {"verified_files": len(manifest["files"]), "raw_bytes": sum((root / s["path"]).stat().st_size for s in spec["sources"]),
                        "scoring_implementation_available": name in {"gsm8k", "processbench"}}
    return result
