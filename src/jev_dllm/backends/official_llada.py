"""CUDA adapter for the pinned official LLaDA sampler; imports are lazy."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import time
from ..schema import Candidate, CostRecord, Question


class OfficialLLaDA:
    def __init__(self, config: dict, sampler_path: Path, prompt: str, allow_download: bool = False):
        import re
        backend = config["backends"]["official_llada"]
        if not all(re.fullmatch(r"[0-9a-f]{40}", config["generator"].get(field) or "")
                   for field in ("model_revision", "tokenizer_revision")):
            raise ValueError("Immutable model and tokenizer revisions are required before loading")
        if prompt.count("{question}") != 1:
            raise ValueError("Generation prompt must contain exactly one {question} placeholder")
        if sha256(sampler_path.read_bytes()).hexdigest() != backend["source_sha256"]:
            raise ValueError("Sampler source does not match the reviewed, pinned file")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise RuntimeError("Install the separate GPU runtime described in docs/runbook.md") from None
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU using CUDA_VISIBLE_DEVICES; no model has been loaded")
        if torch.cuda.get_device_properties(0).major < 8 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native BF16 is required: use an available A100; V100 needs a separately reviewed precision protocol")
        generator = config["generator"]
        spec = importlib.util.spec_from_file_location("jev_pinned_llada_sampler", sampler_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.torch, self.sampler, self.config, self.prompt = torch, module.generate, config, prompt
        options = {"trust_remote_code": True, "local_files_only": not allow_download}
        self.tokenizer = AutoTokenizer.from_pretrained(generator["model_id"],
            revision=generator["tokenizer_revision"], **options)
        self.model = AutoModel.from_pretrained(generator["model_id"],
            revision=generator["model_revision"], torch_dtype=torch.bfloat16, **options).to("cuda").eval()

    def _sample(self, inputs):
        g = self.config["generator"]
        steps = self.config["backends"]["official_llada"]["steps"]
        output = self.sampler(self.model, inputs.input_ids, attention_mask=inputs.attention_mask,
            steps=steps, gen_length=g["generation_length"], block_length=g["block_size"],
            temperature=g["temperature"], cfg_scale=0., remasking="low_confidence")
        return output, steps

    def generate(self, question: Question, *, candidate_id: str, seed: int) -> tuple[Candidate, CostRecord]:
        torch, g = self.torch, self.config["generator"]
        text = self.prompt.replace("{question}", question.prompt)
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                      tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda")
        # Sequential pilot: every candidate has an independent, recorded seed.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        with torch.inference_mode():
            output, nfe = self._sample(inputs)
        end.record()
        torch.cuda.synchronize()
        gpu_seconds = start.elapsed_time(end) / 1000
        ids = output[0, inputs.input_ids.shape[1]:].tolist()
        stop_ids = {126081, 126348}
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)
        stop = next((i for i, token in enumerate(ids) if token in stop_ids), None)
        # No end marker is conservatively flagged; do not silently infer completion.
        completed = 126336 not in ids
        decoded = self.tokenizer.decode(ids[:stop] if stop is not None else ids, skip_special_tokens=True)
        candidate = Candidate(question.question_id, candidate_id, seed, decoded, completed, stop is None)
        return candidate, CostRecord(
            trajectory_nfe=nfe,
            generation_gpu_seconds=gpu_seconds,
            generation_wall_seconds=time.perf_counter() - started,
            peak_memory_bytes=torch.cuda.max_memory_allocated())
