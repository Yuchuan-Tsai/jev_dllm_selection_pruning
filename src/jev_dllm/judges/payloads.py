"""Explicit allowlists: never serialize entire internal experiment records."""
import math
from ..schema import Candidate, PartialDraft, Question


def final_request(question: Question, candidates: list[Candidate]) -> dict:
    if not candidates:
        raise ValueError("No candidates supplied")
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    for c in candidates:
        if c.question_id != question.question_id or not c.completed or c.truncated:
            raise ValueError("Final judging requires completed, nontruncated candidates for this question")
    return {
        "stage": "final", "question": question.prompt,
        "candidates": [{"candidate_id": c.candidate_id, "text": c.text} for c in candidates],
    }


def partial_request(question: Question, drafts: list[PartialDraft]) -> dict:
    if not drafts or len({d.candidate_id for d in drafts}) != len(drafts):
        raise ValueError("Drafts must be nonempty with unique IDs")
    for d in drafts:
        if d.question_id != question.question_id or not 0 <= d.completed_blocks < d.total_blocks:
            raise ValueError("Invalid partial draft progress")
        if not math.isfinite(d.remaining_budget) or d.remaining_budget < 0 or d.budget_unit not in {"trajectory_nfe", "gpu_seconds"}:
            raise ValueError("Invalid remaining budget")
    return {
        "stage": "partial", "question": question.prompt,
        "candidates": [{
            "candidate_id": d.candidate_id, "text": d.text,
            "completed_blocks": d.completed_blocks, "total_blocks": d.total_blocks,
            "remaining_budget": d.remaining_budget, "budget_unit": d.budget_unit,
            "presentation": d.presentation,
        } for d in drafts],
    }
