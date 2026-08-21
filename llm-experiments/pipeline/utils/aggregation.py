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
# IWMV  (Iterative Weighted Majority Vote — previously mislabeled as ISP)
# ---------------------------------------------------------------------------

def run_iwmv(
    models: list[str],
    matrix: dict[str, list[int | None]],
    max_iter: int = 20,
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float], dict]:
    """Returns (hard_labels, accuracies, weights, conv_info).

    conv_info keys:
      n_iters          — iterations until convergence (or max_iter if not converged)
      converged        — whether convergence criterion was met
      weight_range     — max(w) - min(w): spread of final weights across models
      most_changed     — model whose weight changed most from iter-1 to final
      max_weight_delta — magnitude of that change
    """
    n = len(next(iter(matrix.values())))
    mv = _majority_vote(_votes_by_sample(models, matrix))
    soft: list[float] = [0.8 if v == 1 else (0.2 if v == 0 else 0.5) for v in mv]

    accuracies: dict[str, float] = {}
    weights:    dict[str, float] = {}
    init_weights: dict[str, float] = {}
    n_iters_done = max_iter
    converged    = False

    for it in range(max_iter):
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
        if it == 0:
            init_weights = dict(weights)
        for i in range(n):
            score = 0.0
            for m in models:
                v = matrix[m][i]
                if v is None:
                    continue
                score += weights[m] if v == 1 else -weights[m]
            soft[i] = 1.0 / (1.0 + math.exp(-score))
        if max(abs(a - b) for a, b in zip(soft, old_soft)) < eps:
            n_iters_done = it + 1
            converged    = True
            break

    weight_deltas = {m: abs(weights[m] - init_weights.get(m, weights[m])) for m in models}
    most_changed  = max(weight_deltas, key=weight_deltas.get)
    conv_info = {
        "n_iters":          n_iters_done,
        "converged":        converged,
        "weight_range":     round(max(weights.values()) - min(weights.values()), 4),
        "most_changed":     most_changed,
        "max_weight_delta": round(weight_deltas[most_changed], 4),
    }

    hard: list[int | None] = [1 if p > 0.5 else (0 if p < 0.5 else None) for p in soft]
    return hard, accuracies, weights, conv_info


# ---------------------------------------------------------------------------
# ISP  (Inverse Surprising Popularity — Algorithm 2, arXiv:2510.01499)
# ---------------------------------------------------------------------------

def _compute_conditional_probs(
    models: list[str],
    matrix: dict[str, list[int | None]],
    eps: float = 1e-6,
) -> dict[tuple[str, str, int, int], float]:
    """P(judge_i = k | judge_j = l) for every ordered pair (i,j) and label pair (k,l)."""
    n = len(next(iter(matrix.values())))
    cond: dict[tuple[str, str, int, int], float] = {}
    for mi in models:
        for mj in models:
            if mi == mj:
                continue
            for l in (0, 1):
                count_l = count_k1 = 0
                for t in range(n):
                    vi, vj = matrix[mi][t], matrix[mj][t]
                    if vi is None or vj is None:
                        continue
                    if vj == l:
                        count_l += 1
                        if vi == 1:
                            count_k1 += 1
                p1 = count_k1 / count_l if count_l > 0 else 0.5
                cond[(mi, mj, 1, l)] = max(eps, min(1 - eps, p1))
                cond[(mi, mj, 0, l)] = 1.0 - cond[(mi, mj, 1, l)]
    return cond


def run_isp(
    models: list[str],
    matrix: dict[str, list[int | None]],
) -> tuple[list[int | None], dict[tuple[str, str, int, int], float]]:
    """Real ISP for K=2 (Algorithm 2, Ai/Pan et al. arXiv:2510.01499).

    AdvISP(s) = (# judges who voted s) - sum_i mean_{j≠i} P(i says s | j says 1-vote_j)
    Label = argmax_s AdvISP(s).

    Returns (labels, conditional_prob_table).
    Note: no per-model accuracy/weight outputs — ISP aggregates at the item level.
    """
    n    = len(next(iter(matrix.values())))
    cond = _compute_conditional_probs(models, matrix)

    labels: list[int | None] = []
    for t in range(n):
        votes_t = {m: matrix[m][t] for m in models}
        valid   = [m for m in models if votes_t[m] is not None]
        if len(valid) < 2:
            labels.append(None)
            continue

        adv: dict[int, float] = {}
        for s in (0, 1):
            vote_count = sum(1 for m in valid if votes_t[m] == s)
            sisp_sum   = 0.0
            for i in valid:
                others = [j for j in valid if j != i]
                if not others:
                    continue
                # Counterfactual: how often would i predict s if each j had said the opposite?
                sisp_sum += sum(
                    cond.get((i, j, s, 1 - votes_t[j]), 0.5) for j in others
                ) / len(others)
            adv[s] = vote_count - sisp_sum

        if adv[1] > adv[0]:
            labels.append(1)
        elif adv[0] > adv[1]:
            labels.append(0)
        else:
            labels.append(None)

    return labels, cond


# ---------------------------------------------------------------------------
# OWI  (Optimal Weighting Iterated — arXiv:2510.01499)
# ---------------------------------------------------------------------------

def run_owi(
    models: list[str],
    matrix: dict[str, list[int | None]],
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    """Uses real ISP pseudo-labels as reference. Returns (labels, accuracies, weights)."""
    isp_labels, _ = run_isp(models, matrix)
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
