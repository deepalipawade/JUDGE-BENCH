from __future__ import annotations

"""
Alternative aggregation strategies for MEMERAG binary judge outputs.
Reads the existing judgement JSON — no new API calls needed.

Implements two unsupervised methods:

1. OW-I  (Optimal Weighting, Iterated)
   From: "Beyond Majority Voting: LLM Aggregation by Leveraging
          Higher-Order Information" (arXiv:2510.01499)
   For K=2: iteratively estimates per-model accuracy from pseudo-labels
   and applies log-odds weighting.  Provably beats majority voting.

2. Dawid-Skene EM
   The binary-classification analog of BT-σ
   (arXiv:2602.16610, "Who Can We Trust? LLM-as-a-Jury").
   BT-σ learns per-judge reliability from pairwise logit probabilities
   (ranking tasks).  Our task is binary classification with no logit
   access, so Dawid-Skene achieves the same goal via EM:
   jointly estimate per-judge confusion matrices (sensitivity +
   specificity) and the true label distribution — fully unsupervised.

Output keys added per record:
  owi_label         — OW-I aggregated label
  dawid_skene_label — Dawid-Skene aggregated label

Usage:
  python aggregator_weighted.py --lang en
  python aggregator_weighted.py --lang es --exclude gemini-2.5-flash-lite

Edit INPUT_JSON / OUTPUT_JSON before running.
"""

import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import argparse

ROOT = Path(__file__).resolve().parents[2]

SUPPORTED     = "Supported"
NOT_SUPPORTED = "Not Supported"
LABELS        = {SUPPORTED, NOT_SUPPORTED}

# Encode labels as integers for arithmetic
ENCODE = {SUPPORTED: 1, NOT_SUPPORTED: 0}
DECODE = {1: SUPPORTED, 0: NOT_SUPPORTED}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = data.get("records", [])
    return [r for r in records if isinstance(r, dict)]


def extract_votes(
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[int | None]]]:
    """Return (model_list, {model: [encoded_label | None, ...]})."""
    models: list[str] = list(records[0].get("model_outputs", {}).keys())
    votes: dict[str, list[int | None]] = {m: [] for m in models}
    for rec in records:
        for m in models:
            label = rec.get("model_outputs", {}).get(m, {}).get("label")
            votes[m].append(ENCODE.get(label))  # None if missing
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


def compute_bacc(
    labels: list[int | None], gold: list[int | None]
) -> float | None:
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
    pos = tp + fn
    neg = tn + fp
    if pos == 0 or neg == 0:
        return None
    return 0.5 * (tp / pos + tn / neg)


def compute_kappa(
    labels: list[int | None], gold: list[int | None]
) -> float | None:
    agree = total = 0
    pred_1 = pred_0 = gold_1 = gold_0 = 0
    for pred, g in zip(labels, gold):
        if pred is None or g is None:
            continue
        total += 1
        if pred == g:
            agree += 1
        if pred == 1:
            pred_1 += 1
        else:
            pred_0 += 1
        if g == 1:
            gold_1 += 1
        else:
            gold_0 += 1
    if total == 0:
        return None
    po = agree / total
    pe = (gold_1 / total) * (pred_1 / total) + (gold_0 / total) * (pred_0 / total)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


# ---------------------------------------------------------------------------
# Method 1: OW-I  (Beyond Majority Voting, arXiv:2510.01499)
# ---------------------------------------------------------------------------

def owi(
    models: list[str],
    vote_matrix: dict[str, list[int | None]],
    isp_labels: list[int | None],
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    """
    Optimal Weighting (OW-I)  —  arXiv:2510.01499.

    Per the paper, OW-I is NOT itself iterative.  It uses ISP pseudo-labels
    (already computed) as a fixed reference to estimate per-model accuracy,
    then performs a single weighted aggregation pass.

    Algorithm:
      1. Estimate per-model accuracy against ISP pseudo-labels.
      2. Weight_m = logit(accuracy_m) — log-odds of being right.
      3. label = sign of weighted sum of votes  (one pass, no loop).

    Returns: (labels, accuracies, weights)
    """
    n = len(next(iter(vote_matrix.values())))

    # Step 1: estimate per-model accuracy from ISP pseudo-labels
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
        # Paper assumes agents are at least random (≥ 1/K = 0.5); floor here
        accuracies[m] = max(0.5 + eps, min(1.0 - eps, acc))

    # Step 2: log-odds weights
    weights: dict[str, float] = {
        m: math.log(accuracies[m] / (1.0 - accuracies[m])) for m in models
    }

    # Step 3: single weighted vote pass
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
# Method 2: ISP  (Beyond Majority Voting, arXiv:2510.01499)
# ---------------------------------------------------------------------------

def isp(
    models: list[str],
    vote_matrix: dict[str, list[int | None]],
    max_iter: int = 20,
    eps: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    """
    Iterative Soft Pseudo-labeling (ISP).

    Same as OW-I but pseudo-labels stay soft (probabilities) throughout
    rather than being snapped to hard 0/1 after each weighted vote.
    Soft labels preserve uncertainty and reduce error propagation across
    iterations, particularly when judges disagree.

    Algorithm:
      1. Start with soft pseudo-labels from majority vote
         (0.8 if majority=1, 0.2 if majority=0).
      2. Estimate per-model accuracy as expected agreement with soft labels.
      3. Weight_m = logit(accuracy_m).
      4. New soft label = sigmoid of weighted sum of votes.
      5. Repeat from 2 until convergence.
      6. Threshold at 0.5 for final hard label.

    Returns: (hard_labels, accuracies, weights)
    """
    n = len(next(iter(vote_matrix.values())))
    sample_votes = votes_by_sample(models, vote_matrix)

    # Step 1: initialise with soft majority-vote pseudo-labels
    mv = majority_vote(sample_votes)
    soft_pseudo: list[float] = [
        0.8 if v == 1 else (0.2 if v == 0 else 0.5) for v in mv
    ]

    accuracies: dict[str, float] = {}
    weights: dict[str, float] = {}

    for _ in range(max_iter):
        old_soft = list(soft_pseudo)

        # Step 2: soft expected accuracy for each model
        for m in models:
            mv_m = vote_matrix[m]
            num = denom = 0.0
            for i in range(n):
                v = mv_m[i]
                if v is None:
                    continue
                p = soft_pseudo[i]
                # expected agreement = p if vote=1 (truth~=1), (1-p) if vote=0
                num   += p if v == 1 else (1.0 - p)
                denom += 1.0
            acc = num / denom if denom > 0 else 0.5
            accuracies[m] = max(0.51, min(1.0 - eps, acc))

        # Step 3: log-odds weights
        for m in models:
            a = accuracies[m]
            weights[m] = math.log(a / (1.0 - a))

        # Step 4: new soft labels via sigmoid of weighted sum
        for i in range(n):
            score = 0.0
            for m in models:
                v = vote_matrix[m][i]
                if v is None:
                    continue
                # vote=1 adds weight, vote=0 subtracts weight
                score += weights[m] if v == 1 else -weights[m]
            soft_pseudo[i] = 1.0 / (1.0 + math.exp(-score))

        # Step 5: convergence check on soft labels
        delta = max(abs(a - b) for a, b in zip(soft_pseudo, old_soft))
        if delta < eps:
            break

    # Step 6: threshold to hard labels
    hard_labels: list[int | None] = [
        1 if p > 0.5 else (0 if p < 0.5 else None) for p in soft_pseudo
    ]
    return hard_labels, accuracies, weights


# ---------------------------------------------------------------------------
# Method 3: Dawid-Skene EM  (analog of BT-σ for binary classification)
# ---------------------------------------------------------------------------

def dawid_skene(
    models: list[str],
    vote_matrix: dict[str, list[int | None]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[list[int | None], dict[str, float], dict[str, float], dict[str, float]]:
    """
    Dawid-Skene EM for binary labels.

    Per-judge parameters learned (unsupervised):
      alpha_m  = P(model m predicts Supported | truth = Supported)   [sensitivity]
      beta_m   = P(model m predicts Not Supp.  | truth = Not Supp.)  [specificity]

    Relationship to BT-σ:
      BT-σ learns a per-judge discriminator sigma_k that down-weights
      noisy/inconsistent judges in a pairwise ranking setup.
      Here, (alpha_m, beta_m) play the same role: a judge with
      alpha_m ≈ beta_m ≈ 0.5 is pure noise and its influence on the
      posterior P(y=1|x) approaches zero automatically.

    Returns: (labels, alpha, beta, p_y1_per_sample)
    """
    n = len(next(iter(vote_matrix.values())))

    # --- Initialise p(y=1|x) via majority vote ---
    sample_votes = votes_by_sample(models, vote_matrix)
    mv = majority_vote(sample_votes)
    p_y1: list[float] = [
        0.8 if v == 1 else (0.2 if v == 0 else 0.5)
        for v in mv
    ]

    prevalence = sum(p_y1) / n

    # --- Initialise per-model parameters ---
    alpha: dict[str, float] = {m: 0.8 for m in models}  # sensitivity
    beta:  dict[str, float] = {m: 0.8 for m in models}  # specificity

    LOG_EPS = 1e-10

    for _ in range(max_iter):
        old_p_y1 = list(p_y1)

        # ── M-step: update alpha, beta, prevalence ───────────────────────
        for m in models:
            mv_m = vote_matrix[m]

            # alpha_m = E[votes=1 & truth=1] / E[truth=1]
            num_alpha = denom_alpha = 0.0
            # beta_m  = E[votes=0 & truth=0] / E[truth=0]
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

            alpha[m] = num_alpha / denom_alpha if denom_alpha > 0 else 0.8
            beta[m]  = num_beta  / denom_beta  if denom_beta  > 0 else 0.8

            # Clip to [0.5, 0.99] so each judge adds signal, not noise
            alpha[m] = max(0.5, min(0.99, alpha[m]))
            beta[m]  = max(0.5, min(0.99, beta[m]))

        prevalence = sum(p_y1) / n
        prevalence = max(LOG_EPS, min(1.0 - LOG_EPS, prevalence))

        # ── E-step: update p(y=1|x) ──────────────────────────────────────
        for i in range(n):
            log_p1 = math.log(prevalence)
            log_p0 = math.log(1.0 - prevalence)

            for m in models:
                v = vote_matrix[m][i]
                if v is None:
                    continue
                if v == 1:
                    log_p1 += math.log(alpha[m] + LOG_EPS)
                    log_p0 += math.log(1.0 - beta[m]  + LOG_EPS)
                else:
                    log_p1 += math.log(1.0 - alpha[m] + LOG_EPS)
                    log_p0 += math.log(beta[m]  + LOG_EPS)

            # Numerically stable normalisation
            log_max = max(log_p1, log_p0)
            p1 = math.exp(log_p1 - log_max)
            p0 = math.exp(log_p0 - log_max)
            p_y1[i] = p1 / (p1 + p0)

        # Convergence check
        delta = max(abs(a - b) for a, b in zip(p_y1, old_p_y1))
        if delta < tol:
            break

    labels: list[int | None] = [
        1 if p > 0.5 else (0 if p < 0.5 else None)
        for p in p_y1
    ]
    return labels, alpha, beta, p_y1


# ---------------------------------------------------------------------------
# Metrics table printer
# ---------------------------------------------------------------------------

def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.2f}%"


def print_comparison_table(
    gold: list[int | None],
    results: dict[str, list[int | None]],
) -> None:
    headers = ["Method", "Bal.Acc", "Kappa", "Coverage"]
    rows: list[list[str]] = []
    for name, labels in results.items():
        covered = sum(1 for l in labels if l is not None)
        rows.append([
            name,
            pct(compute_bacc(labels, gold)),
            pct(compute_kappa(labels, gold)),
            f"{covered}/{len(labels)}",
        ])
    all_rows = [headers] + rows
    widths = [max(len(r[i]) for r in all_rows) + 2 for i in range(len(headers))]

    def fmt(items: list[str]) -> str:
        return "".join(v.ljust(w) for v, w in zip(items, widths))

    print(fmt(headers))
    print("".join("-" * w for w in widths))
    for row in rows:
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
    try:
        tmp.replace(path)
    except PermissionError:
        import time as _time
        fallback = path.with_suffix(path.suffix + f".{int(_time.time())}.bak")
        tmp.replace(fallback)
        print(f"[WARN] File locked, saved to: {fallback}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OW-I, ISP, Dawid-Skene aggregation for MEMERAG.")
    parser.add_argument("--lang", default="en", help="Language subfolder (en, es, de, hi, fr, ...)")
    parser.add_argument("--exclude", default=None,
                        help="Model ID to exclude from the panel (e.g. gemini-2.5-flash-lite)")
    parser.add_argument(
        "--tiebreak",
        choices=["best_bacc", "insertion"],
        default="best_bacc",
        help=(
            "How to break 2-2 majority-vote ties. "
            "'best_bacc': best individual balanced-accuracy model goes first (default). "
            "'insertion': JSON insertion order (same implicit behaviour as QAGS eval)."
        ),
    )
    args = parser.parse_args()

    INPUT_JSON  = ROOT / "results_tmp" / "memerag_ext" / args.lang / f"memerag_judgement_{args.lang}.json"
    OUTPUT_JSON = ROOT / "results_tmp" / "memerag_ext" / args.lang / f"aggregator_weighted_{args.lang}.json"
    if args.exclude:
        excl_folder = "excl_" + args.exclude.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        OUTPUT_JSON = OUTPUT_JSON.parent / excl_folder / OUTPUT_JSON.name
        print(f"[INFO] Excluding model from panel: {args.exclude}")

    print(f"Input:     {INPUT_JSON}")
    print(f"Output:    {OUTPUT_JSON}")
    print(f"Tiebreak:  {args.tiebreak}\n")

    records = load_records(INPUT_JSON)
    if not records:
        print("[ERROR] No records found.")
        return

    models, vote_matrix = extract_votes(records)
    if args.exclude:
        models = [m for m in models if m != args.exclude]
        vote_matrix = {m: v for m, v in vote_matrix.items() if m != args.exclude}
    n = len(records)

    gold: list[int | None] = [
        ENCODE.get(r.get("gold_label")) for r in records
    ]

    if args.tiebreak == "best_bacc":
        model_baccs = {m: compute_bacc(vote_matrix[m], gold) or 0.0 for m in models}
        models = sorted(models, key=lambda m: model_baccs[m], reverse=True)
        tiebreak_desc = f"best_bacc (first: {models[0].split('/')[-1]})"
    else:
        tiebreak_desc = f"insertion (first: {models[0].split('/')[-1]})"

    print(f"Records : {n}")
    print(f"Models (tiebreak order): {models}")
    gold_dist = Counter(r.get("gold_label") for r in records)
    print(f"Gold distribution: {dict(gold_dist)}\n")

    # Missing label counts per model
    print("Missing labels per model:")
    for m in models:
        n_missing = sum(1 for v in vote_matrix[m] if v is None)
        print(f"  {m}: {n_missing} missing")
    print()

    # ── Majority vote (baseline) ─────────────────────────────────────────
    sample_votes = votes_by_sample(models, vote_matrix)
    mv_labels = majority_vote(sample_votes)

    # ── ISP (runs first — OW-I depends on its pseudo-labels) ────────────
    print("Running ISP...")
    t0 = time.time()
    isp_labels, isp_acc, isp_weights = isp(models, vote_matrix)
    print(f"  Done in {time.time() - t0:.2f}s")
    print("  Per-model accuracy estimates (vs ISP soft pseudo-labels):")
    for m in models:
        print(f"    {m}: acc={isp_acc[m]:.4f}  weight={isp_weights[m]:.4f}")
    print()

    # ── OW-I (single pass using ISP pseudo-labels) ───────────────────────
    print("Running OW-I...")
    t0 = time.time()
    owi_labels, owi_acc, owi_weights = owi(models, vote_matrix, isp_labels)
    print(f"  Done in {time.time() - t0:.2f}s")
    print("  Per-model accuracy estimates (vs ISP pseudo-labels):")
    for m in models:
        print(f"    {m}: acc={owi_acc[m]:.4f}  weight={owi_weights[m]:.4f}")
    print()

    # ── Dawid-Skene ─────────────────────────────────────────────────────
    print("Running Dawid-Skene EM...")
    t0 = time.time()
    ds_labels, ds_alpha, ds_beta, ds_py1 = dawid_skene(models, vote_matrix)
    print(f"  Done in {time.time() - t0:.2f}s")
    print("  Per-model reliability estimates (sensitivity / specificity):")
    for m in models:
        print(f"    {m}: sensitivity(α)={ds_alpha[m]:.4f}  specificity(β)={ds_beta[m]:.4f}")
    print()

    # ── Comparison ───────────────────────────────────────────────────────
    individual: dict[str, list[int | None]] = {
        m: [ENCODE.get(r.get("model_outputs", {}).get(m, {}).get("label")) for r in records]
        for m in models
    }
    all_results: dict[str, list[int | None]] = {
        **individual,
        "panel_majority_vote": mv_labels,
        "owi": owi_labels,
        "isp": isp_labels,
        "dawid_skene": ds_labels,
    }

    print("── Results comparison ──────────────────────────────────────────────")
    print_comparison_table(gold, all_results)

    # ── Save ─────────────────────────────────────────────────────────────
    output_records = []
    for i, rec in enumerate(records):
        row = {
            "sample_id":         rec.get("sample_id"),
            "query_id":          rec.get("query_id"),
            "sentence_id":       rec.get("sentence_id"),
            "gold_label":        rec.get("gold_label"),
            "panel_majority":    rec.get("panel_majority"),
            "model_outputs": {
                m: {"label": DECODE.get(vote_matrix[m][i]), "original_label": rec.get("model_outputs", {}).get(m, {}).get("label")}
                for m in models
            },
            "owi_label":          DECODE.get(owi_labels[i]),
            "isp_label":          DECODE.get(isp_labels[i]),
            "dawid_skene_label":  DECODE.get(ds_labels[i]),
            "dawid_skene_p_supported": round(ds_py1[i], 4),
        }
        output_records.append(row)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_input": str(INPUT_JSON),
        "n_samples": n,
        "tiebreak_strategy": tiebreak_desc,
        "models": models,
        "method_params": {
            "owi": {
                "description": "Optimal Weighting Iterated (Beyond MV, arXiv:2510.01499)",
                "per_model_accuracy": owi_acc,
                "per_model_weight": owi_weights,
            },
            "isp": {
                "description": "Iterative Soft Pseudo-labeling (Beyond MV, arXiv:2510.01499)",
                "per_model_accuracy": isp_acc,
                "per_model_weight": isp_weights,
            },
            "dawid_skene": {
                "description": "Dawid-Skene EM — binary classification analog of BT-σ (arXiv:2602.16610)",
                "per_model_sensitivity_alpha": ds_alpha,
                "per_model_specificity_beta": ds_beta,
            },
        },
        "metrics": {
            name: {
                "balanced_accuracy": pct(compute_bacc(labels, gold)),
                "cohen_kappa":       pct(compute_kappa(labels, gold)),
            }
            for name, labels in all_results.items()
        },
        "records": output_records,
    }

    save_json(OUTPUT_JSON, summary)
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
