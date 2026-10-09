from typing import Protocol
from ..schema import CostRecord


class Judge(Protocol):
    def score(self, request: dict) -> tuple[dict[str, float], CostRecord]:
        """Return candidate-id scores and measured request cost; labels are prohibited."""
        ...

    def choose(self, request: dict) -> tuple[str, CostRecord]:
        """Return one candidate ID from a listwise request."""
        ...
