"""Reference-free, versioned extraction of one complete Python code block."""
import ast
from dataclasses import dataclass
import re

VERSION = "python_single_block_v1"
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\r\n]*)\r?\n?$")


@dataclass(frozen=True)
class Extraction:
    mode: str
    source: str | None
    reason: str | None
    start: int | None
    end: int | None


def extract_python(text: str) -> Extraction:
    """No tests, references or entry-point hints can influence source extraction.

    Valid whole Python is unchanged (including fence literals inside strings).
    Otherwise, accept exactly one pair of matching standalone fences; the opening
    label must be Python/python3/py or empty. Ignore surrounding prose, preserve
    the enclosed character span exactly, and never repair code. Unfenced syntax
    errors remain raw source for the isolated Python worker to classify.
    """
    if not isinstance(text, str):
        raise TypeError("Source must be a string")
    if not text.strip():
        return Extraction("rejected", None, "empty_response", None, None)
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        pass
    else:
        return Extraction("raw", text, None, 0, len(text))
    markers = []
    offset = 0
    for line in text.splitlines(keepends=True):
        match = FENCE.fullmatch(line)
        if match:
            markers.append((offset, offset + len(line), match[1], match[2].strip()))
        offset += len(line)
    if not markers:
        return Extraction("raw", text, None, 0, len(text))
    if len(markers) != 2:
        return Extraction("rejected", None, "ambiguous_or_unclosed_fences", None, None)
    opening, closing = markers
    if (opening[2][0] != closing[2][0] or len(closing[2]) < len(opening[2]) or closing[3]):
        return Extraction("rejected", None, "mismatched_fences", None, None)
    if opening[3].lower() not in {"", "python", "python3", "py"}:
        return Extraction("rejected", None, "unsupported_language", None, None)
    start, end = opening[1], closing[0]
    source = text[start:end]
    if not source.strip():
        return Extraction("rejected", None, "empty_code_block", None, None)
    return Extraction("fenced", source, None, start, end)


def fixtures():
    """Independent examples; no benchmark answers are embedded in this module."""
    good = "def increment(x):\n    return x + 1\n"
    wrong = "def increment(x):\n    return x - 1\n"
    return [
        ("plain", good, "raw", good, "passed"),
        ("python_fence", "```python\n" + good + "```", "fenced", good, "passed"),
        ("unlabeled", "```\n" + good + "```\n", "fenced", good, "passed"),
        ("surrounding_prose", "Here is the function.\n```Python\n" + good + "```\nExplanation.", "fenced", good, "passed"),
        ("tilde_py", "~~~py\n" + good + "~~~", "fenced", good, "passed"),
        ("long_fence", "````python3\n" + good + "`````", "fenced", good, "passed"),
        ("wrong_logic", "```python\n" + wrong + "```", "fenced", wrong, "assertion_failure"),
        ("broken_syntax", "```python\ndef increment(:\n```", "fenced", "def increment(:\n", "syntax_error"),
        ("missing_import", "```python\ndef increment(x): return math.floor(x)+1\n```", "fenced", "def increment(x): return math.floor(x)+1\n", "runtime_error"),
        ("unclosed", "```python\n" + good, "rejected", None, "extraction_failure"),
        ("multiple", "```python\n"+wrong+"```\n```python\n"+good+"```", "rejected", None, "extraction_failure"),
        ("mismatch", "```python\n"+good+"~~~", "rejected", None, "extraction_failure"),
        ("other_language", "```javascript\n"+good+"```", "rejected", None, "extraction_failure"),
        ("empty", "```python\n\n```", "rejected", None, "extraction_failure"),
        ("blank_response", "  \n", "rejected", None, "extraction_failure"),
    ]
