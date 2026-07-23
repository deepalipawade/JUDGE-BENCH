from __future__ import annotations

import re
from collections import Counter
from typing import Any


SUPPORTED     = "Supported"
NOT_SUPPORTED = "Not Supported"
VALID_LABELS  = {SUPPORTED, NOT_SUPPORTED}


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "supported":
        return SUPPORTED
    if text in {"not supported", "unsupported"}:
        return NOT_SUPPORTED
    return None


def normalize_annotation_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def majority_vote_label(values: list[Any]) -> str | None:
    labels = [normalize_label(v) for v in values]
    valid  = [l for l in labels if l in VALID_LABELS]
    if not valid:
        return None
    counts = Counter(valid)
    if counts[SUPPORTED] == counts[NOT_SUPPORTED]:
        return None
    return counts.most_common(1)[0][0]


def extract_label_and_reason(raw_output: str | None) -> tuple[str | None, str | None]:
    """Parse <answer>/<Answer> tag first, then fall back to keyword scan."""
    if not raw_output:
        return None, None

    answer_match = re.search(
        r"<[Aa]nswer>\s*(supported|not supported)\s*</[Aa]nswer>",
        raw_output, flags=re.IGNORECASE,
    )
    if answer_match:
        label = normalize_label(answer_match.group(1))
    else:
        lowered = raw_output.lower()
        if "not supported" in lowered:
            label = NOT_SUPPORTED
        elif "supported" in lowered:
            label = SUPPORTED
        else:
            label = None

    if label is None:
        return None, None

    for tag in ("rationale", "Rationale", "Reasoning", "reasoning"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", raw_output, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return label, m.group(1).strip()

    lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
    return label, " ".join(lines[1:]).strip() if len(lines) > 1 else None
