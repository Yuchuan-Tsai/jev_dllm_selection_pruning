"""Commit-time confidence and checkpoint drafts for the pinned Fast-dLLM sampler.

Recording never draws random numbers and never changes a tensor the sampler
uses, so a recorded trajectory is token-for-token the trajectory an unrecorded
run with the same seed produces. Upstream source stays unmodified: the recorder
wraps the module-level get_transfer_index the sampler looks up at call time and
observes model forwards through a hook.
"""
from dataclasses import replace
import inspect
import math
from .fast_llada import FastLLaDA

MASK_ID = 126336
STOP_IDS = (126081, 126348)


def checkpoint_blocks(total_blocks: int, completed_blocks) -> tuple:
    blocks = tuple(completed_blocks)
    if (not blocks or any(type(b) is not int for b in blocks) or list(blocks) != sorted(set(blocks))
            or not all(0 < b < total_blocks for b in blocks)):
        raise ValueError("Checkpoints must be increasing completed-block counts inside the generation")
    return blocks


class TrajectoryRecorder:
    def __init__(self, sampler, model, torch, checkpoints, total_blocks: int, mask_id: int = MASK_ID):
        # The sampler is wrapped by @torch.no_grad(); its own module globals sit behind __wrapped__.
        self.globals = inspect.unwrap(sampler).__globals__
        self.model, self.torch = model, torch
        self.checkpoints = checkpoint_blocks(total_blocks, checkpoints)
        self.total_blocks, self.mask_id = total_blocks, mask_id
        self.original = self.globals["get_transfer_index"]
        self.handle = None
        self.active = False

    def install(self):
        if self.globals["get_transfer_index"] is not self.original or self.handle is not None:
            raise RuntimeError("Sampler is already instrumented")
        self.globals["get_transfer_index"] = self._transfer
        self.handle = self.model.register_forward_hook(self._forward)

    def remove(self):
        self.globals["get_transfer_index"] = self.original
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def start(self, prompt_length: int, total_length: int):
        self.prompt_length, self.total_length = prompt_length, total_length
        self.forwards = self.block_starts = 0
        self.commits, self.snapshots = [], []
        cuda = self.torch.cuda.is_available() and self.model.device.type == "cuda"
        self.start_event = self.torch.cuda.Event(enable_timing=True) if cuda else None
        if self.start_event is not None:
            self.start_event.record()
        self.active = True

    def _transfer(self, logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
        x0, transfer = self.original(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold)
        if self.active:
            # Same float64 softmax the sampler ranks with, evaluated only at committed positions.
            positions = transfer[0].nonzero().flatten()
            p = self.torch.softmax(logits[0, positions].to(self.torch.float64), dim=-1)
            probability = p.gather(-1, x0[0, positions].unsqueeze(-1)).squeeze(-1)
            offset = self.total_length - x.shape[1] - self.prompt_length
            self.commits.append((positions + offset, probability, self.forwards))
        return x0, transfer

    def _forward(self, module, args, output):
        if not self.active:
            return
        self.forwards += 1
        ids = args[0]
        if ids.shape[1] != self.total_length:
            return
        # Each block begins with one full-length forward; earlier blocks are complete here.
        completed, self.block_starts = self.block_starts, self.block_starts + 1
        if completed in self.checkpoints:
            state = ids[0, self.prompt_length:].clone()
            fill = output.logits[0, self.prompt_length:].argmax(-1)
            event = None
            if self.start_event is not None:
                event = self.torch.cuda.Event(enable_timing=True)
                event.record()
            self.snapshots.append((completed, self.forwards - 1, state,
                                   self.torch.where(state == self.mask_id, fill, state), event))

    def finish(self, answer_length: int) -> dict:
        self.active = False
        length = self.total_length - self.prompt_length
        probability, step = [None] * length, [None] * length
        for positions, values, forward in self.commits:
            for position, value in zip(positions.tolist(), values.tolist()):
                probability[position], step[position] = value, forward
        answer = [p for p in probability[:answer_length] if p is not None]
        confidence = {"mean_probability": None, "min_probability": None, "mean_log_probability": None,
                      "scored_answer_tokens": len(answer)}
        if answer:
            confidence.update(mean_probability=sum(answer) / len(answer), min_probability=min(answer),
                              mean_log_probability=sum(math.log(max(p, 1e-300)) for p in answer) / len(answer))
        checkpoints = [{"completed_blocks": completed, "total_blocks": self.total_blocks, "nfe_before": nfe,
                        "gpu_seconds_before": None if event is None else self.start_event.elapsed_time(event) / 1000,
                        "state_ids": state.tolist(), "argmax_fill_ids": draft.tolist()}
                       for completed, nfe, state, draft, event in self.snapshots]
        return {"mask_id": self.mask_id, "answer_tokens": answer_length, "commit_probability": probability,
                "commit_forward": step, "confidence": confidence, "checkpoints": checkpoints,
                "method": "commit-time float64 softmax recomputed outside the sampler; no random draws"}


class InstrumentedFastLLaDA(FastLLaDA):
    """FastLLaDA whose candidates carry a confidence and whose last_trace holds the trajectory log."""

    def __init__(self, config: dict, root, prompt: str, checkpoints, allow_download: bool = False):
        super().__init__(config, root, prompt, allow_download)
        g = config["generator"]
        self.recorder = TrajectoryRecorder(self.sampler, self.model, self.torch, checkpoints,
                                           g["generation_length"] // g["block_size"])
        self.recorder.install()
        self.last_trace = None

    def _sample(self, inputs):
        prompt_length = inputs.input_ids.shape[1]
        self.recorder.start(prompt_length, prompt_length + self.config["generator"]["generation_length"])
        output, nfe = super()._sample(inputs)
        self.generated_ids = output[0, prompt_length:].tolist()
        return output, nfe

    def generate(self, question, *, candidate_id, seed):
        candidate, cost = super().generate(question, candidate_id=candidate_id, seed=seed)
        # Mirror OfficialLLaDA.generate's stop rule so confidence covers exactly the decoded answer.
        stops = set(STOP_IDS) | ({self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set())
        answer_length = next((i for i, t in enumerate(self.generated_ids) if t in stops), len(self.generated_ids))
        trace = self.recorder.finish(answer_length)
        self.last_trace = {"question_id": question.question_id, "candidate_id": candidate_id, "seed": seed, **trace}
        return replace(candidate, confidence=trace["confidence"]["mean_probability"]), cost
