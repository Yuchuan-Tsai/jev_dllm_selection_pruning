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


SCORERS = ("gsm8k_strict_v1", "gsm8k_units_v2", "gsm8k_units_v3")


def numeric_answer(text: str, scorer: str = "gsm8k_strict_v1") -> Fraction | None:
    """Versioned extraction; v2 uses GSM8K's numeric magnitude convention.

    A percent answer of 25% maps to reference 25, never to .25. Units are
    stripped, not converted. Only explicit final markers/boxes are accepted.
    V3 additionally accepts explicit compound units with a time denominator.
    This is a GSM8K formatting policy, not dimensional or symbolic reasoning.
    """
    if scorer not in SCORERS:
        raise ValueError(f"Unknown numeric scorer: {scorer}")
    answer = extract_final_answer(text)
    if scorer == "gsm8k_strict_v1" or answer is None:
        return numeric_value(answer)
    if len(answer) > 512:
        return None
    answer = answer.strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("$", "$")):
        if answer.startswith(left) and answer.endswith(right) and len(answer) > len(left) + len(right):
            answer = answer[len(left):-len(right)].strip()
            break
    if scorer == "gsm8k_units_v3":
        # Only a trailing unit wrapper is unwrapped; arbitrary LaTeX is not evaluated.
        answer = re.sub(r"\\(?:text|mathrm)\{([A-Za-z %/]+)\}\.?$", r" \1", answer).strip()
    else:
        answer = re.sub(r"\\(?:text|mathrm)\{([A-Za-z %]+)\}$", r" \1", answer).strip()
    answer = answer.replace(r"\%", "%")
    if answer.startswith(("$", "£", "€")):
        answer = answer[1:].strip()
    units = r"(?:%|percent|dollars?|cents?|hours?|minutes?|seconds?|days?|weeks?|years?|meters?|kilometers?|miles?|feet|foot|inches?|pounds?|kilograms?|items?)"
    if scorer == "gsm8k_units_v3":
        numerator = r"(?:dollars?|cents?|hours?|minutes?|seconds?|days?|weeks?|years?|meters?|kilometers?|miles?|feet|foot|inches?|pounds?|kilograms?|items?)"
        denominator = r"(?:seconds?|minutes?|hours?|days?|weeks?|years?)"
        compound = numerator + r"(?:\s*/\s*|\s+per\s+)" + denominator
        units = r"(?:" + compound + "|" + units + ")"
    answer = re.sub(r"\s*" + units + r"\.?$", "", answer, flags=re.I).strip()
    return numeric_value(answer)
