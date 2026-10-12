"""Pinned Fast-dLLM v1 prefix-cache adapter using the original LLaDA weights."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from .official_llada import OfficialLLaDA


def source_files(root: Path, config: dict) -> dict:
    backend = config["backends"]["fast_dllm_v1"]
    path = root / backend["source_manifest_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != backend["source_manifest_sha256"]:
        raise ValueError("Fast-dLLM source manifest differs from its configuration pin")
    manifest = json.loads(path.read_text())
    if manifest["code_commit"] != backend["code_commit"]:
        raise ValueError("Fast-dLLM source revision differs from configuration")
    expected = {"llada/generate.py", "llada/model/__init__.py", "llada/model/modeling_llada.py",
                "llada/model/configuration_llada.py", "LICENSE"}
    if set(manifest["files"]) != expected:
        raise ValueError("Unexpected Fast-dLLM source bundle")
    return manifest


def verify_sources(root: Path, config: dict) -> Path:
    manifest = source_files(root, config)
    base = root / "third_party/fast_dllm_v1"
    for name, expected in manifest["files"].items():
        path = base / name
        components = [base, path, *(base / parent for parent in Path(name).parents)]
        if any(p.is_symlink() for p in components):
            raise ValueError("Fast-dLLM source paths must not contain symlinks")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Fast-dLLM source hash differs: {name}")
    return base / "llada"


def load_upstream(root: Path, config: dict):
    path = verify_sources(root, config)
    # Upstream generate.py uses an absolute `model` import. Keep it confined to
    # this verified directory, and refuse collisions with a different package.
    for name, module in list(sys.modules.items()):
        if name == "model" or name.startswith("model."):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(path.resolve()):
                raise RuntimeError("Another model package is loaded; use a fresh experiment process")
    sys.path.insert(0, str(path))
    try:
        model_module = importlib.import_module("model.modeling_llada")
        spec = importlib.util.spec_from_file_location("jev_pinned_fast_sampler", path / "generate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path))
    return model_module, module


def run_sampler(model, sampler, prompt, *, steps, gen_length, block_length,
                temperature, threshold, mask_id=126336):
    calls = 0

    def count_call(_model, _inputs):
        nonlocal calls
        calls += 1
        # A valid step unmasks at least one token. Bound malformed-model loops.
        if calls > gen_length:
            raise RuntimeError("Fast-dLLM exceeded the generation-length forward-call bound")

    handle = model.register_forward_pre_hook(count_call)
    try:
        output, reported = sampler(model, prompt, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=temperature, remasking="low_confidence",
            threshold=threshold, factor=None, mask_id=mask_id)
    finally:
        handle.remove()
    if type(reported) is not int or reported != calls or calls < 1:
        raise RuntimeError("Fast-dLLM reported NFE differs from measured forward calls")
    return output, calls


class FastLLaDA(OfficialLLaDA):
    def __init__(self, config: dict, root: Path, prompt: str, allow_download: bool = False):
        from ..config import validate
        validate(config)
        if not all(re.fullmatch(r"[0-9a-f]{40}", config["generator"].get(name) or "")
                   for name in ("model_revision", "tokenizer_revision")):
            raise ValueError("Immutable model and tokenizer revisions are required before loading")
        if prompt.count("{question}") != 1:
            raise ValueError("Generation prompt must contain exactly one {question} placeholder")
        verify_sources(root, config)
        import torch
        from transformers import AutoTokenizer
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU")
        if torch.cuda.get_device_properties(0).major < 8 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Fast-dLLM profile requires native BF16")
        model_module, sampler_module = load_upstream(root, config)
        g = config["generator"]
        options = {"revision": g["model_revision"], "local_files_only": not allow_download}
        model_config = model_module.LLaDAConfig.from_pretrained(g["model_id"], **options)
        # Use upstream default PyTorch SDPA; do not add flash-attn or compilation.
        model_config.flash_attention = False
        model, info = model_module.LLaDAModelLM.from_pretrained(g["model_id"],
            config=model_config, torch_dtype=torch.bfloat16, output_loading_info=True, **options)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("Fast-dLLM model does not exactly match the pinned LLaDA checkpoint")
        self.model = model.to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(g["model_id"], revision=g["tokenizer_revision"],
            trust_remote_code=True, local_files_only=not allow_download)
        self.torch, self.config, self.prompt = torch, config, prompt
        self.sampler = sampler_module.generate_with_prefix_cache

    def _sample(self, inputs):
        g, b = self.config["generator"], self.config["backends"]["fast_dllm_v1"]
        if not self.torch.all(inputs.attention_mask == 1):
            raise ValueError("Fast-dLLM pilot supports one unpadded prompt only")
        return run_sampler(self.model, self.sampler, inputs.input_ids, steps=b["steps"],
            gen_length=g["generation_length"], block_length=g["block_size"],
            temperature=g["temperature"], threshold=b["confidence_threshold"])


def check_runtime(root: Path, config: dict) -> dict:
    """Tiny CPU model compatibility check; no model download or benchmark claim."""
    import torch
    import transformers
    model_module, sampler = load_upstream(root, config)
    torch.manual_seed(7)
    # The llama block has a separate up-projection gate. Use SiLU, as in
    # the pinned checkpoint; the default SwiGLU would halve only one branch.
    tiny = model_module.LLaDAConfig(d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
        activation_type="silu",
        vocab_size=32, embedding_size=32, max_sequence_length=32, rope=True,
        weight_tying=False, block_type="llama", flash_attention=False,
        attention_dropout=0., residual_dropout=0., embedding_dropout=0.)
    model = model_module.LLaDAModelLM(tiny, init_params=True).eval()
    prompt = torch.tensor([[1, 2, 3]])
    # Suppress only the synthetic mask ID; exercise the real model/cache path.
    def logits_hook(_model, _inputs, result):
        result.logits[..., 31] = float("-inf")
        return result
    hook = model.register_forward_hook(logits_hook)
    try:
        with torch.inference_mode():
            output, nfe = run_sampler(model, sampler.generate_with_prefix_cache, prompt,
                steps=4, gen_length=4, block_length=2, temperature=1., threshold=.9, mask_id=31)
        if output.shape != (1, 7) or (output[:, 3:] == 31).any() or not torch.equal(output[:, :3], prompt):
            raise RuntimeError("Tiny Fast-dLLM runtime validation failed")
    finally:
        hook.remove()
    return {"status": "passed", "check": "tiny_cpu_prefix_cache", "forward_calls": nfe,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "real_model_validated": False, "model_weights_downloaded": False}
