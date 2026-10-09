"""Conservative fixture/GSM8K-style extraction, NOT a MATH500 equivalence engine."""
import re
from fractions import Fraction


def extract_final_answer(text: str) -> str | None:
    # Parse the last boxed answer using brace depth (nested fractions are allowed).
    start = text.rfind(r"\boxed{")
    if start >= 0:
        start += len(r"\boxed{")
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            if depth == 0:
                return text[start:index].strip() or None
        return None
    markers = list(re.finditer(r"(?:####|Final answer\s*:)\s*([^\n]+)", text, re.I))
    return (markers[-1].group(1).strip() or None) if markers else None


def numeric_value(answer: str | None) -> Fraction | None:
    if answer is None:
        return None
    answer = answer.strip().replace("−", "-")
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    latex = re.fullmatch(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", answer)
    if latex:
        answer = f"{latex[1]}/{latex[2]}"
    number = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    if not re.fullmatch(number + r"(?:/[+-]?\d+)?", answer):
        return None
    try:
        return Fraction(answer.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def score_numeric(text: str, reference: str) -> bool:
    expected = numeric_value(reference)
    if expected is None:
        raise ValueError("Reference is not numeric; configure a task-specific evaluator")
    observed = numeric_value(extract_final_answer(text))
    return observed is not None and observed == expected
