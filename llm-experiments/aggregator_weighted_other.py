from __future__ import annotations

"""
Alternative aggregation strategies (OW-I and Dawid-Skene) for QAGS and DICES.
Reads existing judgement JSONs — no new API calls needed.

Usage:
    python aggregator_weighted_other.py --dataset qags_xsum
    python aggregator_weighted_other.py --dataset qags_xsum --exclude gemini-2.5-flash-lite
    python aggregator_weighted_other.py --dataset qags_cnndm
    python aggregator_weighted_other.py --dataset qags_cnndm --exclude gemini-2.5-flash-lite
    python aggregator_weighted_other.py --dataset dices350
    python aggregator_weighted_other.py --dataset dices350 --exclude gemini-2.5-flash-lite


Edit the OUTPUT paths per dataset in DATASET_CONFIGS below if needed.
"""

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Dataset configs  — edit paths here if your layout differs
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "qags_xsum": {
        "input":        ROOT / "results_tmp" / "qags_xsum" / "qags_judgement.json",
        "output":       ROOT / "results_tmp" / "qags_xsum" / "aggregator_weighted_qags_xsum.json",
        "outputs_key":  "model_outputs",
        "gold_key":     "majority_human",
        "pos_label":    "yes",
        "neg_label":    "no",
        # Pre-computed metrics from verified eval CSVs (avoids JSON alignment issues)
        # Format: {display_name: {"csv": path, "filter_col": col, "filter_val": val,
        #                         "bacc_col": col, "kappa_col": col, "cov_col": col|None}}
        "aggregator_csv_rows": [
            {
                "name":       "Agg (Gemini-Flash-Lite)",
                "csv":        ROOT / "results_tmp" / "qags_xsum" / "gemini_qags_judgement_metrics.csv",
                "filter_col": "kind", "filter_val": "aggregator",
                "bacc_col":   "balanced_accuracy", "kappa_col": "kappa",
                "cov_col":    "total",
            },
            {
                "name":       "Agg (Llama-3.3-70B)",
                "csv":        ROOT / "results_tmp" / "qags_xsum" / "meta_qags_judgement_metrics.csv",
                "filter_col": "kind", "filter_val": "aggregator",
                "bacc_col":   "balanced_accuracy", "kappa_col": "kappa",
                "cov_col":    "total",
            },
        ],
    },
    "qags_cnndm": {
        "input":        ROOT / "results_tmp" / "qags_cnndm" / "qags_judgement.json",
        "output":       ROOT / "results_tmp" / "qags_cnndm" / "aggregator_weighted_qags_cnndm.json",
        "outputs_key":  "model_outputs",
        "gold_key":     "majority_human",
        "pos_label":    "yes",
        "neg_label":    "no",
        "aggregator_csv_rows": [
            {
                "name":       "Agg (Llama-3.3-70B)",
                "csv":        ROOT / "results_tmp" / "qags_cnndm" / "qags_judgement_metrics.csv",
                "filter_col": "kind", "filter_val": "aggregator",
                "bacc_col":   "balanced_accuracy", "kappa_col": "kappa",
                "cov_col":    "total",
            },
        ],
    },
    "dices350": {
        "input":        ROOT / "results_tmp" / "dices350" / "meta_judgement.json",
        "output":       ROOT / "results_tmp" / "dices350" / "aggregator_weighted_dices350.json",
        "outputs_key":  "panel_outputs",
        "gold_key":     "safety_gold",
        "pos_label":    "Yes",
        "neg_label":    "No",
        # DICES eval CSV uses a "method" column with free-text row names
        "aggregator_csv_rows": [
            {
                "name":       "Aggregator",
                "csv":        ROOT / "results_tmp" / "dices350" / "meta_dices_eval.csv",
                "filter_col": "method", "filter_val": "Aggregator",
                "bacc_col":   "balanced_accuracy", "kappa_col": "cohen_kappa",
                "cov_col":    "coverage",
            },
            {
                "name":       "Gemma Aggregator",
                "csv":        ROOT / "results_tmp" / "dices350" / "meta_dices_eval.csv",
                "filter_col": "method", "filter_val": "Gemma aggregator",
                "bacc_col":   "balanced_accuracy", "kappa_col": "cohen_kappa",
                "cov_col":    "coverage",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = data.get("records", []) if isinstance(data, dict) else data
    return [r for r in records if isinstance(r, dict)]


def extract_votes(
    records: list[dict[str, Any]],
    outputs_key: str,
    pos_label: str,
    neg_label: str,
) -> tuple[list[str], dict[str, list[int | None]]]:
    """Return (model_list, {model: [1|0|None, ...]}) where 1=pos, 0=neg."""
    encode = {pos_label.lower(): 1, neg_label.lower(): 0}
    models: list[str] = list(records[0].get(outputs_key, {}).keys())
    votes: dict[str, list[int | None]] = {m: [] for m in models}
    for rec in records:
        for m in models:
            raw_label = rec.get(outputs_key, {}).get(m, {}).get("label")
            encoded = encode.get(str(raw_label).lower()) if raw_label is not None else None
            votes[m].append(encoded)
    return models, votes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def majority_vote(votes_per_sample: list[list[int | None]]) -> list[int | None]:
    results: list[int | None] = []
    for sample_votes in votes_per_sample:
        valid = [v for v in sample_votes if v is not None]
        if not valid:
            results.append(None)
            continue
        c = Counter(valid)
        top = c.most_common()
        if len(top) == 1 or top[0][1] != top[1][1]:
            results.append(top[0][0])
        else:
            # 2-2 tie: first model in list breaks the tie
            results.append(next((v for v in sample_votes if v is not None), None))
    return results


def votes_by_sample(
    models: list[str], vote_matrix: dict[str, list[int | None]]
) -> list[list[int | None]]:
    n = len(next(iter(vote_matrix.values())))
    return [[vote_matrix[m][i] for m in models] for i in range(n)]


def compute_pairwise_kappa(
    a: list[int | None], b: list[int | None]
) -> float | None:
    agree = total = a1 = a0 = b1 = b0 = 0
    for va, vb in zip(a, b):
        if va is None or vb is None:
            continue
        total += 1
        a1 += va; a0 += 1 - va
        b1 += vb; b0 += 1 - vb
        if va == vb:
            agree += 1
    if total == 0:
        return None
    po = agree / total
    pe = (a1 / total) * (b1 / total) + (a0 / total) * (b0 / total)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


def print_interjudge_kappa(
    models: list[str], vote_matrix: dict[str, list[int | None]]
) -> dict[tuple[str, str], float | None]:
    """Print pairwise Cohen's Kappa between all judges and return the matrix."""
    short = {m: m.split("/")[-1][:22] for m in models}
    col_w = max(len(s) for s in short.values()) + 2

    kappa_matrix: dict[tuple[str, str], float | None] = {}
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i < j:
                k = compute_pairwise_kappa(vote_matrix[ma], vote_matrix[mb])
                kappa_matrix[(ma, mb)] = k
                kappa_matrix[(mb, ma)] = k

    print("Inter-judge pairwise Cohen's Kappa:")
    header = f"  {'':46}" + "".join(short[m].ljust(col_w) for m in models)
    print(header)
    print("  " + "-" * (46 + col_w * len(models)))
    for ma in models:
        row = f"  {short[ma]:<46}"
        for mb in models:
            if ma == mb:
                row += "—".ljust(col_w)
            else:
                k = kappa_matrix.get((ma, mb))
                row += (pct(k) if k is not None else "n/a").ljust(col_w)
        print(row)
    print()
    return kappa_matrix


def compute_bacc(labels: list[int | None], gold: list[int | None]) -> float | None:
    tp = tn = fp = fn = 0
    for pred, g in zip(labels, gold):
        if pred is None or g is None:
            continue
        if g == 1 and pred == 1:
            tp += 1
        elif g == 0 and pred == 0:
            tn += 1
        elif g == 0 and pred == 1:
            fp += 1
        elif g == 1 and pred == 0:
            fn += 1
    pos, neg = tp + fn, tn + fp
    if pos == 0 or neg == 0:
        return None
    return 0.5 * (tp / pos + tn / neg)


def compute_kappa(labels: list[int | None], gold: list[int | None]) -> float | None:
    agree = total = pred_1 = pred_0 = gold_1 = gold_0 = 0
    for pred, g in zip(labels, gold):
        if pred is None or g is None:
            continue
        total += 1
        if pred == g:
            agree += 1
        pred_1 += pred
        pred_0 += 1 - pred
        gold_1 += g
        gold_0 += 1 - g
    if total == 0:
        return None
    po = agree / total
    pe = (gold_1 / total) * (pred_1 / total) + (gold_0 / total) * (pred_0 / total)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


# ---------------------------------------------------------------------------
# Tie counting
# ---------------------------------------------------------------------------

def count_ties(models: list[str], vote_matrix: dict[str, list[int | None]]) -> tuple[int, int]:
    """Return (n_ties, n_valid) — how many valid samples have an exact vote tie."""
    n = len(next(iter(vote_matrix.values())))
    ties = valid = 0
    for i in range(n):
        v = [vote_matrix[m][i] for m in models if vote_matrix[m][i] is not None]
        if not v:
            continue
        valid += 1
        c = Counter(v)
        top = c.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
    return ties, valid


# ---------------------------------------------------------------------------
# Simple (prompted) aggregator — read pre-computed metrics from CSV
# ---------------------------------------------------------------------------

def read_aggregator_csv_rows(
    cfg_rows: list[dict[str, Any]],
) -> list[tuple[str, float | None, float | None, str]]:
    """Read pre-computed aggregator metrics from existing eval CSV files.

    Returns list of (display_name, balanced_accuracy, kappa, coverage_str).
    This avoids JSON alignment issues — the CSV values are already verified.
    """
    import csv

    results = []
    for entry in cfg_rows:
        csv_path: Path = entry["csv"]
        if not csv_path.exists():
            print(f"  [INFO] Aggregator CSV not found, skipping: {csv_path.name}")
            continue

        filter_col = entry["filter_col"]
        filter_val = entry["filter_val"]
        bacc_col   = entry["bacc_col"]
        kappa_col  = entry["kappa_col"]
        cov_col    = entry.get("cov_col")

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get(filter_col, "").strip() == filter_val:
                    try:
                        bacc  = float(row[bacc_col])
                        kappa = float(row[kappa_col])
                    except (ValueError, KeyError):
                        bacc = kappa = None
                    cov_str = str(row.get(cov_col, "?")) if cov_col else "?"
                    results.append((entry["name"], bacc, kappa, cov_str))
                    break
            else:
                print(f"  [WARN] Row '{filter_val}' not found in {csv_path.name}")

    return results


# ---------------------------------------------------------------------------
# OW-I  (Beyond Majority Voting, arXiv:2510.01499)
# ---------------------------------------------------------------------------

def isp(
    models: list[str],
    vote_matrix: dict[str, list[int | None]],
    max_iter: int = 20,
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    """Iterative Soft Pseudo-labeling (ISP) — arXiv:2510.01499."""
    n = len(next(iter(vote_matrix.values())))
    sample_votes = votes_by_sample(models, vote_matrix)
    mv = majority_vote(sample_votes)
    soft_pseudo: list[float] = [0.8 if v == 1 else (0.2 if v == 0 else 0.5) for v in mv]

    accuracies: dict[str, float] = {}
    weights: dict[str, float] = {}

    for _ in range(max_iter):
        old_soft = list(soft_pseudo)

        for m in models:
            mv_m = vote_matrix[m]
            num = denom = 0.0
            for i in range(n):
                v = mv_m[i]
                if v is None:
                    continue
                p = soft_pseudo[i]
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
                v = vote_matrix[m][i]
                if v is None:
                    continue
                score += weights[m] if v == 1 else -weights[m]
            soft_pseudo[i] = 1.0 / (1.0 + math.exp(-score))

        if max(abs(a - b) for a, b in zip(soft_pseudo, old_soft)) < eps:
            break

    hard_labels: list[int | None] = [
        1 if p > 0.5 else (0 if p < 0.5 else None) for p in soft_pseudo
    ]
    return hard_labels, accuracies, weights


def owi(
    models: list[str],
    vote_matrix: dict[str, list[int | None]],
    isp_labels: list[int | None],
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    """
    Optimal Weighting (OW-I) — arXiv:2510.01499.

    Uses ISP pseudo-labels as fixed reference to estimate per-model accuracy,
    then performs a single weighted aggregation pass (no outer iteration).
    """
    n = len(next(iter(vote_matrix.values())))

    accuracies: dict[str, float] = {}
    for m in models:
        mv = vote_matrix[m]
        correct = total = 0
        for i in range(n):
            if mv[i] is not None and isp_labels[i] is not None:
                total += 1
                if mv[i] == isp_labels[i]:
                    correct += 1
        acc = correct / total if total > 0 else 0.5
        accuracies[m] = max(0.5 + eps, min(1.0 - eps, acc))

    weights: dict[str, float] = {
        m: math.log(accuracies[m] / (1.0 - accuracies[m])) for m in models
    }

    labels: list[int | None] = []
    for i in range(n):
        score_1 = score_0 = 0.0
        for m in models:
            v = vote_matrix[m][i]
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

def dawid_skene(
    models: list[str],
    vote_matrix: dict[str, list[int | None]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float], list[float]]:
    n = len(next(iter(vote_matrix.values())))
    sample_votes = votes_by_sample(models, vote_matrix)
    mv = majority_vote(sample_votes)
    p_y1: list[float] = [0.8 if v == 1 else (0.2 if v == 0 else 0.5) for v in mv]
    prevalence = sum(p_y1) / n
    alpha: dict[str, float] = {m: 0.8 for m in models}
    beta:  dict[str, float] = {m: 0.8 for m in models}
    LOG_EPS = 1e-10

    for _ in range(max_iter):
        old_p_y1 = list(p_y1)

        for m in models:
            mv_m = vote_matrix[m]
            num_alpha = denom_alpha = 0.0
            num_beta  = denom_beta  = 0.0
            for i in range(n):
                v = mv_m[i]
                if v is None:
                    continue
                py1 = p_y1[i]
                py0 = 1.0 - py1
                denom_alpha += py1
                denom_beta  += py0
                if v == 1:
                    num_alpha += py1
                else:
                    num_beta  += py0
            alpha[m] = max(0.5, min(0.99, num_alpha / denom_alpha if denom_alpha > 0 else 0.8))
            beta[m]  = max(0.5, min(0.99, num_beta  / denom_beta  if denom_beta  > 0 else 0.8))

        prevalence = max(LOG_EPS, min(1.0 - LOG_EPS, sum(p_y1) / n))

        for i in range(n):
            log_p1 = math.log(prevalence)
            log_p0 = math.log(1.0 - prevalence)
            for m in models:
                v = vote_matrix[m][i]
                if v is None:
                    continue
                if v == 1:
                    log_p1 += math.log(alpha[m] + LOG_EPS)
                    log_p0 += math.log(1.0 - beta[m] + LOG_EPS)
                else:
                    log_p1 += math.log(1.0 - alpha[m] + LOG_EPS)
                    log_p0 += math.log(beta[m] + LOG_EPS)
            log_max = max(log_p1, log_p0)
            p1 = math.exp(log_p1 - log_max)
            p0 = math.exp(log_p0 - log_max)
            p_y1[i] = p1 / (p1 + p0)

        if max(abs(a - b) for a, b in zip(p_y1, old_p_y1)) < tol:
            break

    labels: list[int | None] = [1 if p > 0.5 else (0 if p < 0.5 else None) for p in p_y1]
    return labels, alpha, beta, p_y1


# ---------------------------------------------------------------------------
# Metrics table printer
# ---------------------------------------------------------------------------

def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.2f}%"


def print_comparison_table(
    gold: list[int | None],
    results: dict[str, list[int | None]],
    precomputed: list[tuple[str, float | None, float | None, str]] | None = None,
) -> None:
    """Print comparison table.

    precomputed: list of (name, bacc, kappa, coverage_str) read from existing CSVs.
    These are shown below a separator so it's clear they come from separate eval scripts.
    """
    headers = ["Method", "Bal.Acc", "Kappa", "Coverage"]
    rows: list[list[str]] = []
    for name, labels in results.items():
        covered = sum(1 for l in labels if l is not None)
        rows.append([name, pct(compute_bacc(labels, gold)), pct(compute_kappa(labels, gold)), f"{covered}/{len(labels)}"])

    pre_rows: list[list[str]] = []
    if precomputed:
        for name, bacc, kappa, cov_str in precomputed:
            pre_rows.append([f"{name} [from csv]", pct(bacc), pct(kappa), cov_str])

    all_rows = [headers] + rows + pre_rows
    widths = [max(len(r[i]) for r in all_rows) + 2 for i in range(len(headers))]

    def fmt(items: list[str]) -> str:
        return "".join(v.ljust(w) for v, w in zip(items, widths))

    print(fmt(headers))
    print("".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))
    if pre_rows:
        print("".join("·" * w for w in widths) + "  ← simple aggregators (pre-computed from eval CSVs)")
        for row in pre_rows:
            print(fmt(row))
    print()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OW-I and Dawid-Skene aggregation for QAGS and DICES.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Which dataset to run: qags_xsum | qags_cnndm | dices350",
    )
    parser.add_argument("--exclude", default=None,
                        help="Model ID to exclude from the panel (e.g. gemini-2.5-flash-lite)")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    INPUT_JSON:  Path = cfg["input"]
    OUTPUT_JSON: Path = cfg["output"]
    if args.exclude:
        _suf = args.exclude.split("/")[-1].replace(".", "_")
        OUTPUT_JSON = OUTPUT_JSON.with_name(OUTPUT_JSON.stem + f"_excl_{_suf}.json")
        print(f"[INFO] Excluding model from panel: {args.exclude}")
    outputs_key: str  = cfg["outputs_key"]
    gold_key:    str  = cfg["gold_key"]
    pos_label:   str  = cfg["pos_label"]
    neg_label:   str  = cfg["neg_label"]

    ENCODE = {pos_label.lower(): 1, neg_label.lower(): 0}
    DECODE = {1: pos_label, 0: neg_label}

    print(f"Dataset : {args.dataset}")
    print(f"Input   : {INPUT_JSON}")
    print(f"Output  : {OUTPUT_JSON}\n")

    if not INPUT_JSON.exists():
        print(f"[ERROR] Input file not found: {INPUT_JSON}")
        return

    records = load_records(INPUT_JSON)
    if not records:
        print("[ERROR] No records found.")
        return

    models, vote_matrix = extract_votes(records, outputs_key, pos_label, neg_label)
    if args.exclude:
        models = [m for m in models if m != args.exclude]
        vote_matrix = {m: v for m, v in vote_matrix.items() if m != args.exclude}
    n = len(records)

    gold: list[int | None] = [
        ENCODE.get(str(r.get(gold_key, "")).lower()) for r in records
    ]

    # Preserve insertion order for the random tiebreaker baseline.
    original_models = list(models)

    # Sort models by individual balanced accuracy (descending) so the best judge
    # for this specific dataset breaks 2-way ties — not a hardcoded order.
    model_baccs = {m: compute_bacc(vote_matrix[m], gold) or 0.0 for m in models}
    models = sorted(models, key=lambda m: model_baccs[m], reverse=True)

    print(f"Records : {n}")
    print(f"Models (tiebreak=random/insertion): {original_models}")
    print(f"Models (tiebreak=best bacc first) : {models}")
    gold_dist = Counter(r.get(gold_key) for r in records)
    print(f"Gold distribution ({gold_key}): {dict(gold_dist)}\n")

    print("Missing labels per model:")
    for m in models:
        n_missing = sum(1 for v in vote_matrix[m] if v is None)
        print(f"  {m}: {n_missing} missing")
    print()

    # ── Majority vote — two tiebreaker variants ──────────────────────────────
    # random: original JSON insertion order (reproducible baseline, no gold needed)
    mv_random_labels = majority_vote(votes_by_sample(original_models, vote_matrix))
    # best: highest individual bacc breaks ties (informed strategy — requires gold)
    mv_best_labels   = majority_vote(votes_by_sample(models, vote_matrix))

    n_ties, n_valid = count_ties(original_models, vote_matrix)
    print(f"Tie samples (exact split among valid votes): {n_ties} / {n_valid} "
          f"({100 * n_ties / n_valid:.1f}%)\n")

    # ── ISP (runs first — OW-I depends on its pseudo-labels) ─────────────────
    print("Running ISP...")
    t0 = time.time()
    isp_labels, isp_acc, isp_weights = isp(models, vote_matrix)
    print(f"  Done in {time.time() - t0:.2f}s")
    print("  Per-model accuracy estimates (vs ISP soft pseudo-labels):")
    for m in models:
        print(f"    {m}: acc={isp_acc[m]:.4f}  weight={isp_weights[m]:.4f}")
    print()

    # ── OW-I (single pass using ISP pseudo-labels) ────────────────────────────
    print("Running OW-I...")
    t0 = time.time()
    owi_labels, owi_acc, owi_weights = owi(models, vote_matrix, isp_labels)
    print(f"  Done in {time.time() - t0:.2f}s")
    print("  Per-model accuracy estimates (vs ISP pseudo-labels):")
    for m in models:
        print(f"    {m}: acc={owi_acc[m]:.4f}  weight={owi_weights[m]:.4f}")
    print()

    # ── Dawid-Skene ──────────────────────────────────────────────────────────
    print("Running Dawid-Skene EM...")
    t0 = time.time()
    ds_labels, ds_alpha, ds_beta, ds_py1 = dawid_skene(models, vote_matrix)
    print(f"  Done in {time.time() - t0:.2f}s")
    print("  Per-model reliability estimates (sensitivity / specificity):")
    for m in models:
        print(f"    {m}: sensitivity(α)={ds_alpha[m]:.4f}  specificity(β)={ds_beta[m]:.4f}")
    print()

    # ── Simple (prompted) aggregators — pre-computed from verified eval CSVs ──
    agg_csv_rows = read_aggregator_csv_rows(cfg.get("aggregator_csv_rows", []))

    # ── Comparison ───────────────────────────────────────────────────────────
    individual: dict[str, list[int | None]] = {
        m: [ENCODE.get(str(r.get(outputs_key, {}).get(m, {}).get("label", "")).lower()) for r in records]
        for m in models
    }
    all_results: dict[str, list[int | None]] = {
        **individual,
        "panel_majority_random": mv_random_labels,
        "panel_majority_best":   mv_best_labels,
        "isp":                   isp_labels,
        "owi":                   owi_labels,
        "dawid_skene":           ds_labels,
    }

    print("── Results comparison ──────────────────────────────────────────────")
    print_comparison_table(gold, all_results, precomputed=agg_csv_rows)

    print("── Inter-judge pairwise kappa (individual judges only) ─────────────")
    kappa_matrix = print_interjudge_kappa(models, vote_matrix)

    # All-method pairwise kappa (individual judges + MV + OW-I + ISP + DS)
    all_method_kappa: dict[str, float | None] = {}
    all_names = list(all_results.keys())
    for i, name_a in enumerate(all_names):
        for name_b in all_names[i + 1:]:
            k = compute_pairwise_kappa(all_results[name_a], all_results[name_b])
            all_method_kappa[f"{name_a} vs {name_b}"] = round(k, 4) if k is not None else None

    # ── Save ─────────────────────────────────────────────────────────────────
    output_records = []
    for i, rec in enumerate(records):
        row = {
            "sample_id":            rec.get("sample_id"),
            "gold_label":           rec.get(gold_key),
            outputs_key: {
                m: {
                    "label":          DECODE.get(vote_matrix[m][i]),
                    "original_label": rec.get(outputs_key, {}).get(m, {}).get("label"),
                }
                for m in models
            },
            "mv_random_label":      DECODE.get(mv_random_labels[i]),
            "mv_best_label":        DECODE.get(mv_best_labels[i]),
            "isp_label":            DECODE.get(isp_labels[i]),
            "owi_label":            DECODE.get(owi_labels[i]),
            "dawid_skene_label":    DECODE.get(ds_labels[i]),
            "dawid_skene_p_pos":    round(ds_py1[i], 4),
        }
        output_records.append(row)

    summary = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "dataset":      args.dataset,
        "source_input": str(INPUT_JSON),
        "n_samples":              n,
        "models_best_tiebreak":   models,
        "models_random_tiebreak": original_models,
        "method_params": {
            "isp": {
                "description":        "Iterative Soft Pseudo-labeling (Beyond MV, arXiv:2510.01499)",
                "per_model_accuracy": isp_acc,
                "per_model_weight":   isp_weights,
            },
            "owi": {
                "description":        "Optimal Weighting — uses ISP pseudo-labels, single pass (Beyond MV, arXiv:2510.01499)",
                "per_model_accuracy": owi_acc,
                "per_model_weight":   owi_weights,
            },
            "dawid_skene": {
                "description":              "Dawid-Skene EM — binary classification analog of BT-σ (arXiv:2602.16610)",
                "per_model_sensitivity_alpha": ds_alpha,
                "per_model_specificity_beta":  ds_beta,
            },
        },
        "metrics": {
            **{
                name: {
                    "balanced_accuracy": pct(compute_bacc(labels, gold)),
                    "cohen_kappa":       pct(compute_kappa(labels, gold)),
                }
                for name, labels in all_results.items()
            },
            # Simple aggregators — pre-computed from eval CSVs (not re-derived here)
            **{
                f"{name} [csv]": {"balanced_accuracy": pct(bacc), "cohen_kappa": pct(kappa)}
                for name, bacc, kappa, _ in agg_csv_rows
            },
        },
        "interjudge_kappa": {
            f"{ma.split('/')[-1]} vs {mb.split('/')[-1]}": round(k, 4) if k is not None else None
            for (ma, mb), k in kappa_matrix.items()
            if ma < mb
        },
        "method_pairwise_kappa": all_method_kappa,
        "records": output_records,
    }

    save_json(OUTPUT_JSON, summary)
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
