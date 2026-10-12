"""Validate the draft Week 1 protocol without asserting experiment readiness."""
import json
import math
from pathlib import Path


PIN_FIELDS = (
    ("generator", "model_revision"),
    ("generator", "tokenizer_revision"),
    ("judge", "model_version"),
    ("judge", "api_contract_version"),
    ("datasets", "manifest_sha256"),
    ("evaluation", "math_equivalence_revision"),
    ("evaluation", "humaneval_harness_revision"),
    ("evaluation", "processbench_revision"),
    ("runtime", "hardware"),
    ("runtime", "dependency_lock_sha256"),
)


def validate(config: dict) -> list[str]:
    if config.get("schema_version") == 2:
        return validate_pilot(config)
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if type(config.get("frozen")) is not bool:
        raise ValueError("frozen must be a boolean")
    generator = config["generator"]
    if generator["model_id"] != "GSAI-ML/LLaDA-8B-Instruct":
        raise ValueError("Primary protocol requires LLaDA-8B-Instruct")
    if generator["dtype"] != "bfloat16":
        raise ValueError("Primary generator precision must be bfloat16")
    for field in ("generation_length", "block_size", "candidates_per_question", "microbatch_size"):
        if type(generator[field]) is not int or generator[field] <= 0:
            raise ValueError(f"generator.{field} must be a positive integer")
    if generator["generation_length"] % generator["block_size"]:
        raise ValueError("generation_length must be divisible by block_size")
    if generator["candidates_per_question"] != 8:
        raise ValueError("Week 1 diversity pilot requires N=8")
    pending = []
    for section, key in PIN_FIELDS:
        value = config[section].get(key)
        if not isinstance(value, str) or not value.strip() or value in {"main", "latest", "TBD"}:
            pending.append(f"{section}.{key}")
    if generator.get("selected_backend") not in {"official_llada", "fast_dllm_v1"}:
        pending.append("generator.selected_backend (choose after the pilot)")
    for backend in ("official_llada", "fast_dllm_v1"):
        values = config["backends"][backend]
        if not values.get("code_commit"):
            pending.append(f"backends.{backend}.code_commit")
        steps = values.get("steps")
        if steps is None:
            pending.append(f"backends.{backend}.steps")
        elif type(steps) is not int or steps <= 0:
            raise ValueError(f"backends.{backend}.steps must be a positive integer")
    temperature = generator.get("temperature")
    if temperature is None:
        pending.append("generator.temperature (verify N=8 diversity on dev)")
    elif not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Week 1 stochastic pilot requires finite positive temperature")
    for name in ("generation_math", "generation_code", "jev_final"):
        prompt = config["prompts"][name]
        if not prompt.get("sha256"):
            pending.append(f"prompts.{name}.sha256")
    if not config["datasets"].get("development_sources"):
        pending.append("datasets.development_sources")
    if config["frozen"] and pending:
        raise ValueError("Cannot mark an unresolved protocol frozen: " + ", ".join(pending))
    return pending


def validate_pilot(config: dict) -> list[str]:
    """Validate the smaller development preflight without replacing the PDF protocol."""
    import re
    if config.get("scope") != "completed_answer_selection" or type(config.get("frozen")) is not bool:
        raise ValueError("Expected a completed-answer selection pilot with a boolean frozen field")
    g = config["generator"]
    if g["model_id"] != "GSAI-ML/LLaDA-8B-Instruct" or g["dtype"] != "bfloat16":
        raise ValueError("Pilot uses fixed LLaDA-8B-Instruct in BF16")
    selected = g.get("selected_backend")
    if selected not in {"official_llada", "fast_dllm_v1"} or g.get("microbatch_size") != 1:
        raise ValueError("The development pilot requires a supported backend and microbatch size 1")
    for field in ("generation_length", "block_size", "candidates_per_question", "microbatch_size"):
        if type(g[field]) is not int or g[field] <= 0:
            raise ValueError(f"generator.{field} must be a positive integer")
    if g["candidates_per_question"] not in (2, 4, 8, 16) or g["generation_length"] % g["block_size"]:
        raise ValueError("Invalid candidate count or block size")
    if type(g["temperature"]) not in (int, float) or not math.isfinite(g["temperature"]) or g["temperature"] <= 0:
        raise ValueError("The diversity pilot requires a finite positive temperature")
    backend = config["backends"][selected]
    steps = backend["steps"]
    if type(steps) is not int or not 0 < steps <= g["generation_length"] or steps % (g["generation_length"] // g["block_size"]):
        raise ValueError("Invalid denoising-step schedule")
    pending = []
    for label, value, pattern in [
        ("generator.model_revision", g["model_revision"], r"[0-9a-f]{40}"),
        ("generator.tokenizer_revision", g["tokenizer_revision"], r"[0-9a-f]{40}"),
        (f"backends.{selected}.code_commit", backend["code_commit"], r"[0-9a-f]{40}"),
        (f"backends.{selected}.source_pin", backend.get("source_sha256") if selected == "official_llada"
             else backend.get("source_manifest_sha256"), r"[0-9a-f]{64}"),
        ("judge.model_version", config["judge"]["model_version"], r"jev-\d+\.\d+\.\d+"),
        ("datasets.manifest_sha256", config["datasets"]["manifest_sha256"], r"[0-9a-f]{64}"),
    ]:
        if value is None:
            pending.append(label)
        elif not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"Invalid pin: {label}")
    if selected == "fast_dllm_v1":
        threshold = backend.get("confidence_threshold")
        if (backend.get("cache") != "prefix" or type(threshold) not in (int, float)
                or not math.isfinite(threshold) or not 0 < threshold <= 1):
            raise ValueError("Fast-dLLM profile requires prefix cache and threshold in (0, 1]")
        if backend.get("source_manifest_path") != "configs/week01/fast-source.json":
            raise ValueError("Fast-dLLM source manifest must use its fixed project path")
    for name in ("generation_math", "jev_final"):
        if not re.fullmatch(r"[0-9a-f]{64}", config["prompts"][name].get("sha256") or ""):
            pending.append(f"prompts.{name}.sha256")
    for name in ("hardware", "dependency_lock_sha256"):
        if not config["runtime"].get(name):
            pending.append(f"runtime.{name}")
    for name in ("generation_validated", "jev_access_validated", "protocol_reviewed"):
        if type(config["readiness"].get(name)) is not bool:
            raise ValueError(f"readiness.{name} must be boolean")
        if not config["readiness"][name]:
            pending.append(f"readiness.{name}")
    if config["frozen"] and pending:
        raise ValueError("Cannot freeze an unresolved pilot: " + ", ".join(pending))
    return pending


def load(path: Path) -> tuple[dict, list[str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return config, validate(config)
