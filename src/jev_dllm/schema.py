"""Public inference records. Reference answers belong only in evaluation/."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Question:
    question_id: str
    dataset: str
    split: Literal["dev", "heldout", "fixture"]
    prompt: str


@dataclass(frozen=True)
class Candidate:
    question_id: str
    candidate_id: str
    seed: int
    text: str
    completed: bool
    truncated: bool
    confidence: float | None = None


@dataclass(frozen=True)
class PartialDraft:
    question_id: str
    candidate_id: str
    text: str
    completed_blocks: int
    total_blocks: int
    remaining_budget: float
    budget_unit: Literal["trajectory_nfe", "gpu_seconds"]
    presentation: Literal["raw_masks", "fully_predicted"]


@dataclass(frozen=True)
class CostRecord:
    # None means unmeasured; never silently convert it to zero.
    trajectory_nfe: int | None = None
    generation_gpu_seconds: float | None = None
    generation_wall_seconds: float | None = None
    draft_preparation_seconds: float | None = None
    judge_request_seconds: float | None = None
    restoration_seconds: float | None = None
    end_to_end_seconds: float | None = None
    peak_memory_bytes: int | None = None
    api_cost_usd: float | None = None
