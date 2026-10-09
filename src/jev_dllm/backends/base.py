from typing import Protocol
from ..schema import Candidate, CostRecord, Question


class GeneratorBackend(Protocol):
    def generate(self, question: Question, *, candidate_id: str, seed: int) -> tuple[Candidate, CostRecord]:
        """Generate using the frozen config; seed each trajectory independently."""
        ...


# Real official_llada and fast_dllm_v1 adapters are intentionally not registered yet.
# Do not use a transformers autoregressive generate() call as a diffusion sampler.
