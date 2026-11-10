"""Pinned Qwen3-14B autoregressive policy adapter for Week 4 development pools."""
import hashlib
import json
from pathlib import Path
import re
import time

from ..schema import Candidate, CostRecord, Question


class Qwen3Policy:
    def __init__(self, policy: dict, allow_download: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU")
        props = torch.cuda.get_device_properties(0)
        if props.major < 8 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Week 4 Qwen policy requires one native BF16 GPU")
        revision = policy["revision"]
        if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
            raise ValueError("Qwen model revision must be an immutable commit identifier")
        options = {"revision": revision, "local_files_only": not allow_download}
        self.tokenizer = AutoTokenizer.from_pretrained(policy["model_id"], **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            policy["model_id"], torch_dtype=torch.bfloat16, **options).to("cuda").eval()
        self.torch, self.policy = torch, policy
        self.gpu = {"name": props.name, "memory_bytes": props.total_memory}

    def generate(self, question: Question, prompt_template: str, *, candidate_id: str,
                 seed: int, max_new_tokens: int) -> tuple[Candidate, CostRecord]:
        torch = self.torch
        user_text = prompt_template.replace("{question}", question.prompt).replace("{problem}", question.prompt)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.policy["thinking"])
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
        sampling = self.policy["sampling"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=sampling["do_sample"],
                temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                pad_token_id=self.tokenizer.eos_token_id)
        end.record()
        torch.cuda.synchronize()
        elapsed_gpu = start.elapsed_time(end) / 1000
        ids = output[0, inputs.input_ids.shape[1]:].tolist()
        completed = bool(ids and ids[-1] == self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        candidate = Candidate(question.question_id, candidate_id, seed, text, completed, not completed)
        return candidate, CostRecord(
            trajectory_nfe=None, generation_gpu_seconds=elapsed_gpu,
            generation_wall_seconds=time.perf_counter() - started,
            peak_memory_bytes=torch.cuda.max_memory_allocated())


def profile(root: Path, config_path: str = "configs/week04/qwen3-14b-policy.json") -> dict:
    path = root / config_path
    config = json.loads(path.read_text())
    if config.get("scope") != "week04_stronger_policy_development":
        raise ValueError("Unexpected Week 4 Qwen policy profile")
    for task, spec in config["policy"]["tasks"].items():
        questions, prompt = root / spec["questions"], root / spec["prompt"]
        if hashlib.sha256(questions.read_bytes()).hexdigest() != spec["questions_sha256"]:
            raise ValueError(f"Pinned development questions changed for {task}")
        if len(json.loads(questions.read_text())) != spec["question_count"]:
            raise ValueError(f"Unexpected development question count for {task}")
        if hashlib.sha256(prompt.read_bytes()).hexdigest() != spec["prompt_sha256"]:
            raise ValueError(f"Generation prompt changed for {task}")
        if "{question}" not in prompt.read_text() and "{problem}" not in prompt.read_text():
            raise ValueError(f"Prompt has no question placeholder for {task}")
    return config
