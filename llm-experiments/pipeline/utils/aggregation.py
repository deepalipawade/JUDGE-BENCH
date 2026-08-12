"""
Pure aggregation algorithms — no API calls, no file I/O.

Each function takes a vote_matrix and returns labels + per-model stats.

Vote encoding: Supported=1, Not Supported=0, missing=None
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any



SUPPORTED     = "Supported"
NOT_SUPPORTED = "Not Supported"
ENCODE = {SUPPORTED: 1, NOT_SUPPORTED: 0}
DECODE = {1: SUPPORTED, 0: NOT_SUPPORTED}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_vote_matrix(
    records: list[dict[str, Any]],
    models: list[str],
    exclude: str | None = None,
) -> tuple[list[str], dict[str, list[int | None]]]:
    """Return (model_list, {model: [encoded_label | None, ...]})."""
    active = [m for m in models if m != exclude]
    matrix: dict[str, list[int | None]] = {m: [] for m in active}
    for rec in records:
        outputs = rec.get("model_outputs", {})
        for m in active:
            label = outputs.get(m, {}).get("label") if isinstance(outputs.get(m), dict) else None
            matrix[m].append(ENCODE.get(label))
    return active, matrix


def gold_labels(records: list[dict[str, Any]]) -> list[int | None]:
    return [ENCODE.get(r.get("gold_label")) for r in records]


def _votes_by_sample(
    models: list[str], matrix: dict[str, list[int | None]]
) -> list[list[int | None]]:
    n = len(next(iter(matrix.values())))
    return [[matrix[m][i] for m in models] for i in range(n)]


def _majority_vote(votes_per_sample: list[list[int | None]]) -> list[int | None]:
    result: list[int | None] = []
    for sample_votes in votes_per_sample:
        valid = [v for v in sample_votes if v is not None]
        if not valid:
            result.append(None)
            continue
        c = Counter(valid)
        top = c.most_common()
        if len(top) == 1 or top[0][1] != top[1][1]:
            result.append(top[0][0])
        else:
            result.append(next((v for v in sample_votes if v is not None), None))
    return result


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def run_majority(
    models: list[str],
    matrix: dict[str, list[int | None]],
) -> list[int | None]:
    return _majority_vote(_votes_by_sample(models, matrix))


# ---------------------------------------------------------------------------
# ISP  (Iterative Soft Pseudo-labeling — arXiv:2510.01499)
# ---------------------------------------------------------------------------

def run_isp(
    models: list[str],
    matrix: dict[str, list[int | None]],
    max_iter: int = 20,
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    """Returns (hard_labels, accuracies, weights)."""
    n = len(next(iter(matrix.values())))
    mv = _majority_vote(_votes_by_sample(models, matrix))
    soft: list[float] = [0.8 if v == 1 else (0.2 if v == 0 else 0.5) for v in mv]

    accuracies: dict[str, float] = {}
    weights:    dict[str, float] = {}

    for _ in range(max_iter):
        old_soft = list(soft)
        for m in models:
            num = denom = 0.0
            for i in range(n):
                v = matrix[m][i]
                if v is None:
                    continue
                p = soft[i]
                num   += p if v == 1 else (1.0 - p)
                denom += 1.0
            acc = num / denom if denom > 0 else 0.5
            accuracies[m] = max(0.51, min(1.0 - eps, acc))
        for m in models:
            a = accuracies[m]
            weights[m] = math.log(a / (1.0 - a))
        for i in range(n):
            score = 0.0
            for m in models:
                v = matrix[m][i]
                if v is None:
                    continue
                score += weights[m] if v == 1 else -weights[m]
            soft[i] = 1.0 / (1.0 + math.exp(-score))
        if max(abs(a - b) for a, b in zip(soft, old_soft)) < eps:
            break

    hard: list[int | None] = [1 if p > 0.5 else (0 if p < 0.5 else None) for p in soft]
    return hard, accuracies, weights


# ---------------------------------------------------------------------------
# OWI  (Optimal Weighting Iterated — arXiv:2510.01499)
# ---------------------------------------------------------------------------

def run_owi(
    models: list[str],
    matrix: dict[str, list[int | None]],
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    """Uses ISP pseudo-labels as reference. Returns (labels, accuracies, weights)."""
    isp_labels, _, _ = run_isp(models, matrix)
    n = len(next(iter(matrix.values())))

    accuracies: dict[str, float] = {}
    for m in models:
        correct = total = 0
        for i in range(n):
            if matrix[m][i] is not None and isp_labels[i] is not None:
                total += 1
                if matrix[m][i] == isp_labels[i]:
                    correct += 1
        acc = correct / total if total > 0 else 0.5
        accuracies[m] = max(0.5 + eps, min(1.0 - eps, acc))

    weights = {m: math.log(accuracies[m] / (1.0 - accuracies[m])) for m in models}

    labels: list[int | None] = []
    for i in range(n):
        score_1 = score_0 = 0.0
        for m in models:
            v = matrix[m][i]
            if v is None:
                continue
            if v == 1:
                score_1 += weights[m]
            else:
                score_0 += weights[m]
        if score_1 > score_0:
            labels.append(1)
        elif score_0 > score_1:
            labels.append(0)
        else:
            labels.append(None)

    return labels, accuracies, weights


# ---------------------------------------------------------------------------
# Dawid-Skene EM
# ---------------------------------------------------------------------------

def run_dawid_skene(
    models: list[str],
    matrix: dict[str, list[int | None]],
    max_iter: int = 100,
    tol: float = 1e-6,
    init_p_y1: list[float] | None = None,
) -> tuple[list[int | None], dict[str, float], dict[str, float], list[float]]:
    """Returns (labels, alpha, beta, p_y1_per_sample)."""
    n = len(next(iter(matrix.values())))
    if init_p_y1 is not None:
        p_y1: list[float] = list(init_p_y1)
    else:
        mv = _majority_vote(_votes_by_sample(models, matrix))
        p_y1 = [0.8 if v == 1 else (0.2 if v == 0 else 0.5) for v in mv]
    prevalence = sum(p_y1) / n
    alpha: dict[str, float] = {m: 0.8 for m in models}
    beta:  dict[str, float] = {m: 0.8 for m in models}
    LOG_EPS = 1e-10

    for _ in range(max_iter):
        old_p_y1 = list(p_y1)

        for m in models:
            na = da = nb = db = 0.0
            for i in range(n):
                v = matrix[m][i]
                if v is None:
                    continue
                py1 = p_y1[i]; py0 = 1.0 - py1
                da += py1; db += py0
                if v == 1:
                    na += py1
                else:
                    nb += py0
            alpha[m] = max(0.5, min(0.99, na / da if da > 0 else 0.8))
            beta[m]  = max(0.5, min(0.99, nb / db if db > 0 else 0.8))

        prevalence = max(LOG_EPS, min(1.0 - LOG_EPS, sum(p_y1) / n))

        for i in range(n):
            lp1 = math.log(prevalence)
            lp0 = math.log(1.0 - prevalence)
            for m in models:
                v = matrix[m][i]
                if v is None:
                    continue
                if v == 1:
                    lp1 += math.log(alpha[m] + LOG_EPS)
                    lp0 += math.log(1.0 - beta[m] + LOG_EPS)
                else:
                    lp1 += math.log(1.0 - alpha[m] + LOG_EPS)
                    lp0 += math.log(beta[m] + LOG_EPS)
            lm = max(lp1, lp0)
            p1 = math.exp(lp1 - lm); p0 = math.exp(lp0 - lm)
            p_y1[i] = p1 / (p1 + p0)

        if max(abs(a - b) for a, b in zip(p_y1, old_p_y1)) < tol:
            break

    labels: list[int | None] = [1 if p > 0.5 else (0 if p < 0.5 else None) for p in p_y1]
    return labels, alpha, beta, p_y1
