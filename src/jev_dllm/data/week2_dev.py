"""Week 2 development sets, disjoint from every held-out and diagnostic input."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import warnings

from .benchmarks import encoded, fetch as fetch_sources, source_rows, verify_raw, write_unchanged
from .splits import _prompt_key
from ..evaluation.answers import extract_final_answer, numeric_value
from ..reproducibility import fingerprint
from ..schema import Question

CATALOG = "configs/week02/dev-sources.json"
MANIFEST = "data/manifests/week02-dev.json"
TASKS = ("gsm8k", "math", "mbpp")


class ConversionError(ValueError):
    """An MBPP task that cannot be rewritten into a HumanEval-style prompt unambiguously."""


def load_catalog(root: Path) -> dict:
    catalog = json.loads((root / CATALOG).read_text())
    if catalog.get("scope") != "week02_development_pools" or set(catalog["tasks"]) != set(TASKS):
        raise ValueError("Unexpected Week 2 development catalog")
    for task in TASKS:
        if not re.fullmatch(r"[0-9a-f]{40}", catalog["tasks"][task]["revision"]):
            raise ValueError("Development sources require immutable Git revisions")
    return catalog


def fetch(root: Path, catalog: dict):
    fetch_sources(root, {"benchmarks": {task: {"sources": [spec["source"]]} for task, spec in catalog["tasks"].items()}})


def ordered(keys, seed, revision, task):
    # Selection depends only on source IDs, never on answers or model outcomes.
    return sorted(keys, key=lambda key: fingerprint([seed, revision, "week02-dev", task, key]))


def protected_prompts(root: Path, catalog: dict) -> set:
    protected = set()
    for relative in catalog["protected_public_inputs"]:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Prepare Week 1 inputs first; missing protected input {relative}")
        for row in json.loads(path.read_text()):
            protected.add(_prompt_key(row.get("prompt") or row["problem"]))
    return protected


def gsm8k_items(rows, spec, excluded_ids):
    for index in range(len(rows)):
        qid = f"gsm8k/train/{index:05}"
        if qid in excluded_ids:
            continue
        row = rows[index]
        answer = numeric_value(extract_final_answer(row["answer"]))
        if answer is None:
            raise ValueError(f"Invalid numeric reference: {qid}")
        yield qid, Question(qid, "gsm8k", "dev", row["question"]), {"question_id": qid, "answer": str(answer)}


def math_items(rows, spec):
    counts = {}
    for row in rows:
        counts[row["unique_id"]] = counts.get(row["unique_id"], 0) + 1
    for row in rows:
        uid = row["unique_id"]
        # Duplicated IDs or empty answers cannot be referenced unambiguously.
        if not uid.startswith(spec["unique_id_prefix"]) or counts[uid] != 1 or not str(row["answer"]).strip():
            continue
        qid = "math/" + uid
        yield qid, Question(qid, "math", "dev", row["problem"]), {
            "question_id": qid, "answer": row["answer"], "subject": row["subject"], "level": row["level"]}


def defined_names(tree) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def convert_mbpp(row) -> tuple[str, dict]:
    """Return a HumanEval-style prompt and reference; raise ConversionError if ambiguous."""
    try:
        # Upstream code uses legacy escapes such as '\w'; those warnings are not conversion errors.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            code, tests = ast.parse(row["code"]), ast.parse("\n".join(row["test_list"]))
    except SyntaxError as exc:
        raise ConversionError("unparseable_reference_or_tests") from exc
    functions = {n.name: n for n in code.body if isinstance(n, ast.FunctionDef)}
    called = {n.func.id for n in ast.walk(tests) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    targets = sorted(name for name in functions if name in called)
    if len(targets) != 1:
        raise ConversionError("ambiguous_or_missing_entry_point")
    entry = targets[0]
    used = {n.id for n in ast.walk(tests) if isinstance(n, ast.Name)}
    if (used & defined_names(code)) - {entry}:
        # Tests would need reference-only classes or helpers the prompt does not supply.
        raise ConversionError("tests_use_other_reference_definitions")
    function = functions[entry]
    signature = f"def {entry}({ast.unparse(function.args)})"
    if function.returns is not None:
        signature += f" -> {ast.unparse(function.returns)}"
    text = " ".join(row["prompt"].split())
    example = row["test_list"][0].strip()
    if '"""' in text + example or "\n" in example:
        raise ConversionError("docstring_unsafe_text")
    body = f"{text}\n\n    Example:\n    {example}"
    # A raw docstring shows backslashes (e.g. regex examples) exactly as the tests use them.
    quote = 'r"""' if "\\" in body else '"""'
    prompt = f'{signature}:\n    {quote}{body}\n    """\n'
    lines = ["def check(candidate):"] + [f"    {line.strip()}" for line in row["test_imports"]]
    lines += [f"    {entry} = candidate"] + [f"    {line.strip()}" for test in row["test_list"] for line in test.splitlines()]
    hidden_tests = "\n".join(lines) + "\n"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(prompt + "    pass\n", "<prompt>", "exec")
            compile(hidden_tests, "<hidden_tests>", "exec")
    except SyntaxError as exc:
        raise ConversionError("converted_text_does_not_compile") from exc
    return prompt, {"entry_point": entry, "hidden_tests": hidden_tests, "canonical_solution": row["code"]}


def mbpp_items(rows, spec, dropped):
    for row in sorted(rows, key=lambda r: r["task_id"]):
        if row["task_id"] < spec["min_task_id"]:
            continue
        qid = f"mbpp/{row['task_id']}"
        try:
            prompt, reference = convert_mbpp(row)
        except ConversionError as exc:
            dropped.append({"question_id": qid, "reason": str(exc)})
            continue
        yield qid, Question(qid, "mbpp", "dev", prompt), {"question_id": qid, **reference}


def select(items, count, seed, revision, task, protected):
    by_id = {qid: (question, reference) for qid, question, reference in items}
    chosen, overlaps, prompts = [], [], set()
    for qid in ordered(by_id, seed, revision, task):
        question, reference = by_id[qid]
        key = _prompt_key(question.prompt)
        if not question.prompt.strip() or key in prompts:
            continue
        if key in protected:
            overlaps.append(qid)
            continue
        prompts.add(key)
        chosen.append((question, reference))
        if len(chosen) == count:
            return chosen, overlaps
    raise ValueError(f"Not enough eligible {task} development questions")


def prepare(root: Path, catalog: dict):
    protected = protected_prompts(root, catalog)
    pilot = {q["question_id"] for q in json.loads((root / "data/processed/gsm8k-dev.json").read_text())}
    count, seed = catalog["count_per_task"], catalog["seed"]
    outputs, tasks = {}, {}
    for task in TASKS:
        spec = catalog["tasks"][task]
        rows = source_rows(root, spec["source"])
        dropped = []
        if task == "gsm8k":
            items = gsm8k_items(rows, spec, pilot)
        elif task == "math":
            items = math_items(rows, spec)
        else:
            items = mbpp_items(rows, spec, dropped)
        chosen, overlaps = select(list(items), count, seed, spec["revision"], task, protected)
        outputs[f"data/processed/{task}-w2dev.json"] = [asdict(q) for q, _ in chosen]
        outputs[f"data/private/{task}-w2dev-references.json"] = [r for _, r in chosen]
        tasks[task] = {"repository": spec["repository"], "revision": spec["revision"], "source": spec["source"],
                       "question_count": len(chosen), "question_ids": [q.question_id for q, _ in chosen],
                       "excluded_for_protected_prompt_overlap": overlaps, "dropped_during_conversion": dropped}
    raw = {path: encoded(value) for path, value in outputs.items()}
    manifest = {"scope": catalog["scope"], "catalog_sha256": hashlib.sha256((root / CATALOG).read_bytes()).hexdigest(),
                "seed": seed, "split": "dev",
                "selection": "lowest_sha256_of_seed_revision_task_source_id_after_eligibility_filters",
                "protected_public_inputs": catalog["protected_public_inputs"], "tasks": tasks,
                "files": {path: hashlib.sha256(value).hexdigest() for path, value in raw.items()},
                "limitation": "Exact normalized prompt overlap is excluded; semantic near-duplicates are not reviewed. "
                              "MBPP is a HumanEval-style proxy, not HumanEval development data."}
    raw[MANIFEST] = encoded(manifest)
    # Check every destination before writing any file.
    for path, value in raw.items():
        if (root / path).exists() and (root / path).read_bytes() != value:
            raise ValueError(f"Prepared development input changed: {path}")
    for path, value in raw.items():
        write_unchanged(root / path, value)
    return audit(root, catalog)


def audit(root: Path, catalog: dict) -> dict:
    for spec in catalog["tasks"].values():
        verify_raw(root / spec["source"]["path"], spec["source"])
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest["catalog_sha256"] != hashlib.sha256((root / CATALOG).read_bytes()).hexdigest():
        raise ValueError("Development catalog changed after preparation")
    for relative, digest in manifest["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared file differs from manifest: {relative}")
    protected = protected_prompts(root, catalog)
    for task in TASKS:
        questions = json.loads((root / f"data/processed/{task}-w2dev.json").read_text())
        if any(_prompt_key(q["prompt"]) in protected or q["split"] != "dev" for q in questions):
            raise ValueError(f"{task} development set overlaps a protected input")
    return {task: {"questions": info["question_count"], "dropped": len(info["dropped_during_conversion"]),
                   "protected_overlaps_excluded": len(info["excluded_for_protected_prompt_overlap"])}
            for task, info in manifest["tasks"].items()}


def mbpp_reference_checks(root: Path):
    """Inputs for running each canonical MBPP solution through the isolated harness."""
    references = json.loads((root / "data/private/mbpp-w2dev-references.json").read_text())
    return [(r["question_id"], r["canonical_solution"], r) for r in references]
