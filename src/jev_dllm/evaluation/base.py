from dataclasses import dataclass
from typing import Protocol
from ..schema import Candidate


@dataclass(frozen=True)
class Reference:
    question_id: str
    answer: str | None = None
    hidden_tests: str | None = None
    entry_point: str | None = None


class Evaluator(Protocol):
    def score(self, candidate: Candidate, reference: Reference) -> bool:
        ...


class Math500Evaluator:
    def score(self, candidate: Candidate, reference: Reference) -> bool:
        raise NotImplementedError("Pin and validate a mathematical equivalence implementation before MATH500 scoring")


class HumanEvalEvaluator:
    def score(self, candidate: Candidate, reference: Reference) -> bool:
        raise NotImplementedError("Connect a tested, isolated HumanEval worker with time/resource limits; do not exec generated code here")
