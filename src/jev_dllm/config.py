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


def load(path: Path) -> tuple[dict, list[str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return config, validate(config)
