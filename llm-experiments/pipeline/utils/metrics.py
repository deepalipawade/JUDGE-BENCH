from __future__ import annotations

from typing import Any

VALID_LABELS = {"Supported", "Not Supported"}


def kappa_from_lists(a: list[str | None], b: list[str | None]) -> float | None:
    """Cohen's Kappa between two parallel label lists."""
    agree = total = 0
    ca: dict[str, int] = {}
    cb: dict[str, int] = {}
    for la, lb in zip(a, b):
        if la not in VALID_LABELS or lb not in VALID_LABELS:
            continue
        total += 1
        ca[la] = ca.get(la, 0) + 1
        cb[lb] = cb.get(lb, 0) + 1
        if la == lb:
            agree += 1
    if total == 0:
        return None
    po = agree / total
    pe = sum((ca.get(l, 0) / total) * (cb.get(l, 0) / total) for l in VALID_LABELS)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


def compute_full_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    """Full confusion matrix metrics — matches evaluate_memerag_results.py format."""
    total = tp = tn = fp = fn = 0
    gold_sup = gold_not = pred_sup = pred_not = 0
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get(pred_key)
        if gold not in VALID_LABELS or pred not in VALID_LABELS:
            continue
        total += 1
        if gold == "Supported":
            gold_sup += 1
        else:
            gold_not += 1
        if pred == "Supported":
            pred_sup += 1
        else:
            pred_not += 1
        if gold == "Supported" and pred == "Supported":
            tp += 1
        elif gold == "Not Supported" and pred == "Not Supported":
            tn += 1
        elif gold == "Not Supported" and pred == "Supported":
            fp += 1
        else:
            fn += 1
    bacc = kappa = accuracy = None
    if total > 0:
        accuracy = (tp + tn) / total
        if gold_sup > 0 and gold_not > 0:
            bacc = 0.5 * (tp / gold_sup + tn / gold_not)
        po = (tp + tn) / total
        pe = (gold_sup / total) * (pred_sup / total) + (gold_not / total) * (pred_not / total)
        kappa = (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)
    return {
        "total_valid": total, "gold_supported": gold_sup, "gold_not_supported": gold_not,
        "pred_supported": pred_sup, "pred_not_supported": pred_not,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "balanced_accuracy": bacc, "cohen_kappa": kappa,
    }


def compute_bacc(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    pos_total = pos_correct = neg_total = neg_correct = 0
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get(pred_key)
        if gold not in VALID_LABELS or pred not in VALID_LABELS:
            continue
        if gold == "Supported":
            pos_total += 1
            if pred == "Supported":
                pos_correct += 1
        else:
            neg_total += 1
            if pred == "Not Supported":
                neg_correct += 1
    if pos_total == 0 or neg_total == 0:
        return None
    return 0.5 * (pos_correct / pos_total + neg_correct / neg_total)


def compute_cohen_kappa(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    n = agree = gold_sup = gold_not = pred_sup = pred_not = 0
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get(pred_key)
        if gold not in VALID_LABELS or pred not in VALID_LABELS:
            continue
        n += 1
        gold_sup += gold == "Supported"
        gold_not += gold == "Not Supported"
        pred_sup += pred == "Supported"
        pred_not += pred == "Not Supported"
        agree    += gold == pred
    if n == 0:
        return None
    po = agree / n
    pe = (gold_sup / n) * (pred_sup / n) + (gold_not / n) * (pred_not / n)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


def pct(v: float | None) -> float | None:
    """Convert 0-1 metric to rounded percentage: 0.8397... → 83.97"""
    return round(v * 100, 2) if v is not None else None


def fmt(v: float | None) -> str:
    """Format for display: 83.97 or n/a"""
    return f"{pct(v):.2f}" if v is not None else "n/a"


def print_metrics_table(rows: list[tuple[str, float | None, float | None]]) -> None:
    """rows: list of (model_name, bacc, kappa)"""
    col_w = max((len(name) for name, _, _ in rows), default=20)
    col_w = max(col_w, 20)
    print(f"\n  {'Model':<{col_w}}  {'Bal. Acc':>10}  {'Kappa':>8}")
    print(f"  {'-'*col_w}  {'-'*10}  {'-'*8}")
    for name, bacc, kappa in rows:
        print(f"  {name:<{col_w}}  {fmt(bacc):>10}  {fmt(kappa):>8}")
