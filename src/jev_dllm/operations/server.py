"""One-GPU BF16 checks, private credentials, and bounded storage planning."""
import json
import os
from pathlib import Path
import re
import shutil


MODEL_WEIGHT_BYTES = 16031162368


def gpu_environment(root: Path):
    device = os.environ.get("GPU_ID", "")
    if not re.fullmatch(r"[0-9]+", device):
        raise ValueError("Set GPU_ID to exactly one available physical GPU index; do not assume GPU0 is free")
    private_home = Path.home() / ".cache/jev-dllm/huggingface"
    shared = os.environ.get("JEV_SHARED_HUB_CACHE", "/shared/rsaas/huggingface/hub")
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": device, "HF_HOME": str(private_home),
                "HF_TOKEN_PATH": str(private_home / "token"), "HF_HUB_CACHE": shared,
                "HF_MODULES_CACHE": str(private_home / "modules"),
                "HF_DATASETS_CACHE": str(root / ".cache/datasets"),
                "HF_XET_CACHE": str(private_home / "xet"),
                "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false"})
    return env


def gpu_check(root: Path):
    # Must run in a child after CUDA_VISIBLE_DEVICES and the cache variables are set.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This workflow requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if properties.major < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 is required. Use an available A100; V100 needs a separately reviewed precision protocol")
    cache = Path(os.environ["HF_HUB_CACHE"])
    if not cache.is_dir():
        raise RuntimeError(f"Shared Hub cache is missing: {cache}. Confirm the lab mount and path before downloading")
    if not os.access(cache, os.R_OK | os.X_OK):
        raise RuntimeError("Shared Hub cache is not readable")
    return {"visible_gpu_count": 1, "gpu_name": properties.name,
            "gpu_total_memory_GiB": round(properties.total_memory / 2**30, 2),
            "gpu_free_memory_GiB": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
            "native_bf16": True, "hub_cache": str(cache),
            "cache_free_GB": round(shutil.disk_usage(cache).free / 1e9, 2),
            "project_filesystem_free_GB": round(shutil.disk_usage(root).free / 1e9, 2),
            "note": "These are hardware/filesystem checks, not quota, scheduler allocation, or experiment validation."}


def fetch_model(root: Path, config_path: Path):
    gpu_check(root)
    from huggingface_hub import snapshot_download
    from ..config import load
    config, _ = load(config_path)
    generator = config["generator"]
    if not re.fullmatch(r"[0-9a-f]{40}", generator["model_revision"]):
        raise ValueError("Model download requires an immutable revision")
    path = snapshot_download(generator["model_id"], revision=generator["model_revision"],
        allow_patterns=["*.safetensors", "*.json", "*.py", "tokenizer.*"])
    return {"cached_model": path, "revision": generator["model_revision"]}


def storage_report(root: Path, candidates=8, questions=1164, runs=10):
    if any(type(v) is not int or v <= 0 for v in (candidates, questions, runs)):
        raise ValueError("Storage scenario counts must be positive integers")
    usage = {}
    for relative in ("data/raw", "data/processed", "data/private", "data/manifests", "artifacts", ".venv", ".venv-gpu"):
        path = root / relative
        usage[relative] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) if path.exists() else 0
    # A planning allowance, not a measurement: 8 KiB/candidate times four for requests/reports/redundancy.
    per_run = questions * candidates * 8192 * 4
    return {"units": "GB=10^9 bytes; GiB=2^30 bytes; disk estimates are separate from GPU VRAM",
            "measured_local_logical_bytes": usage,
            "fixed_model_weight_bytes_from_pinned_index": MODEL_WEIGHT_BYTES,
            "scenario": {"generation_questions": questions, "candidates_per_question": candidates,
                         "runs": runs, "allowance_bytes_per_candidate_including_logs": 32768,
                         "estimated_output_bytes_per_run": per_run, "estimated_output_bytes_all_runs": per_run * runs},
            "initial_pilot_disk_plan_GB": {
                "scope": "One pinned model in personal scratch, one GPU environment, 20-question N=4/N=8 pilots",
                "model_snapshot_approx": 16.1, "gpu_environment_estimate_range": [8, 15],
                "bootstrap_code_small_caches_allowance": 1,
                "steady_use_rounded_estimate_range": [25, 32],
                "temporary_install_extra_allowance_range": [5, 10],
                "conservative_peak_allowance_range": [35, 45],
                "total_personal_server_reservation_including_model": 50,
                "additional_lab_shared_cache_reservation": 0,
                "note": "Planning estimates, not measurements or enforced limits; measure on the server with gpu-disk"},
            "recommended_reservations_GB": {"personal_server_total_including_model": 50,
                                            "local_results_archive": 20},
            "limitations": ["No model copies or optimizer states included in outputs", "No saved full logits/KV tensors",
                            "Personal model cache is included once in the 50 GB pilot reservation; filesystem free space is not personal quota",
                            "Logical file sizes are not filesystem allocated blocks; shared cache is not recursively scanned"]}
