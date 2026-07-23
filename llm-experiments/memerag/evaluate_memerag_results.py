from __future__ import annotations

"""
Evaluate MEMERAG judgement and aggregator results.
Computes OW-I, ISP, and Dawid-Skene internally — no pre-step needed.

Usage:
    python evaluate_memerag_results.py --lang en
    python evaluate_memerag_results.py --lang en --exclude gemini-2.5-flash-lite
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _build_paths(lang: str, excl_folder: str | None = None) -> tuple[Path, Path, Path, Path]:
    base = ROOT / "results_tmp" / "memerag_ext" / lang
    out_dir = base / excl_folder if excl_folder else base
    return (
        base / f"memerag_judgement_{lang}.json",
        out_dir / f"aggregator_gemma_{lang}.json",
        out_dir / f"aggregator_weighted_{lang}.json",
        out_dir / f"random_aggregator_{lang}.json",
    )

LABELS = {"Supported", "Not Supported"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return [r for r in data["records"] if isinstance(r, dict)]
    except Exception:
        pass

    # Partial-file fallback: extract completed objects from the records array
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
    except Exception:
        return []

    records: list[dict[str, Any]] = []
    idx = content.find('"records"')
    if idx == -1:
        return records
    arr_start = content.find("[", idx)
    if arr_start == -1:
        return records
    i = arr_start + 1
    n = len(content)
    while i < n:
        while i < n and content[i] in " \t\r\n,":
            i += 1
        if i >= n or content[i] != "{":
            break
        start, depth = i, 0
        while i < n:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    try:
                        records.append(json.loads(content[start:i]))
                    except Exception:
                        pass
                    break
            i += 1
    return records


def load_meta(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def kappa_from_lists(a: list[str | None], b: list[str | None]) -> float | None:
    """Cohen's Kappa between two parallel lists of string labels."""
    agree = total = 0
    ca: Counter = Counter()
    cb: Counter = Counter()
    for la, lb in zip(a, b):
        if la not in LABELS or lb not in LABELS:
            continue
        total += 1
        ca[la] += 1
        cb[lb] += 1
        if la == lb:
            agree += 1
    if total == 0:
        return None
    po = agree / total
    pe = sum((ca[l] / total) * (cb[l] / total) for l in LABELS)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


def compute_metrics(
    records: list[dict[str, Any]],
    pred_getter,
    gold_key: str = "gold_label",
) -> dict[str, Any]:
    total = tp = tn = fp = fn = 0
    gold_sup = gold_not = pred_sup = pred_not = 0

    for row in records:
        gold = row.get(gold_key)
        pred = pred_getter(row)
        if gold not in LABELS or pred not in LABELS:
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
        "total_valid": total,
        "gold_supported": gold_sup,
        "gold_not_supported": gold_not,
        "pred_supported": pred_sup,
        "pred_not_supported": pred_not,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": bacc,
        "cohen_kappa": kappa,
    }


# ---------------------------------------------------------------------------
# Inter-judge agreement
# ---------------------------------------------------------------------------

def compute_pairwise_kappa(
    records: list[dict[str, Any]],
    model_a: str,
    model_b: str,
) -> float | None:
    labels = ["Supported", "Not Supported"]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    agree = total = 0
    a_counts: Counter = Counter()
    b_counts: Counter = Counter()
    for rec in records:
        outputs = rec.get("model_outputs", {})
        la = outputs.get(model_a, {}).get("label")
        lb = outputs.get(model_b, {}).get("label")
        if la not in label_to_idx or lb not in label_to_idx:
            continue
        total += 1
        a_counts[la] += 1
        b_counts[lb] += 1
        if la == lb:
            agree += 1
    if total == 0:
        return None
    po = agree / total
    pe = sum((a_counts[l] / total) * (b_counts[l] / total) for l in labels)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


def print_interjudge_table(
    records: list[dict[str, Any]], models: list[str]
) -> list[tuple[str, str, float | None]]:
    """Print pairwise kappa matrix and return list of (model_a, model_b, kappa) pairs."""
    pairs: list[tuple[str, str, float | None]] = []
    # Compute all pairs
    kappa_matrix: dict[tuple[str, str], float | None] = {}
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i < j:
                k = compute_pairwise_kappa(records, ma, mb)
                kappa_matrix[(ma, mb)] = k
                kappa_matrix[(mb, ma)] = k
                pairs.append((ma, mb, k))

    # Short names for display
    short = {m: m.split("/")[-1][:18] for m in models}
    col_w = max(len(s) for s in short.values()) + 2

    print("Inter-judge pairwise Cohen's Kappa:")
    header = f"  {'':42}" + "".join(short[m].ljust(col_w) for m in models)
    print(header)
    print("  " + "-" * (42 + col_w * len(models)))
    for ma in models:
        row = f"  {short[ma]:<42}"
        for mb in models:
            if ma == mb:
                row += "—".ljust(col_w)
            else:
                k = kappa_matrix.get((ma, mb))
                row += (pct(k) if k is not None else "n/a").ljust(col_w)
        print(row)
    print()
    return pairs


# ---------------------------------------------------------------------------
# Response analysis
# ---------------------------------------------------------------------------

def print_response_analysis(records: list[dict[str, Any]]) -> None:
    models: dict[str, dict[str, int]] = {}
    for rec in records:
        for model, info in rec.get("model_outputs", {}).items():
            if model not in models:
                models[model] = {"attempts": 0, "success": 0, "null": 0, "error": 0}
            if not isinstance(info, dict):
                continue
            models[model]["attempts"] += 1
            if info.get("error"):
                models[model]["error"] += 1
            elif info.get("label") is None:
                models[model]["null"] += 1
            else:
                models[model]["success"] += 1

    total = len(records)
    print(f"Panel response analysis  ({total} total records)")
    print(f"  {'Model':<42} {'Attempts':>9} {'Success':>9} {'Null':>8} {'Error':>8}")
    print("  " + "-" * 80)
    for model, c in models.items():
        print(f"  {model:<42} {c['attempts']:>9} {c['success']:>9} {c['null']:>8} {c['error']:>8}")
    print()


def print_aggregator_analysis(agg_records: list[dict[str, Any]], model_name: str) -> None:
    success = null = error = 0
    for rec in agg_records:
        if rec.get("aggregator_error"):
            error += 1
        elif rec.get("aggregator_label") is None:
            null += 1
        else:
            success += 1
    total = len(agg_records)
    print(f"Aggregator response analysis  ({total} records)  model={model_name}")
    print(f"  Success: {success}   Null(empty): {null}   Error: {error}")
    print()


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def print_metrics_table(title: str, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
    print(title)
    headers = ["Kind", "Name", "Bal.Acc", "Kappa", "Accuracy", "Valid", "GoldSup", "GoldNot", "TP", "TN", "FP", "FN"]
    table: list[list[str]] = []
    for kind, name, s in rows:
        table.append([
            kind, name,
            pct(s["balanced_accuracy"]),
            pct(s["cohen_kappa"]),
            pct(s["accuracy"]),
            str(s["total_valid"]),
            str(s["gold_supported"]),
            str(s["gold_not_supported"]),
            str(s["tp"]), str(s["tn"]), str(s["fp"]), str(s["fn"]),
        ])
    all_rows = [headers] + table
    widths = [max(len(r[i]) for r in all_rows) + 2 for i in range(len(headers))]

    def fmt(items: list[str]) -> str:
        return "".join(v.ljust(w) for v, w in zip(items, widths))

    print(fmt(headers))
    print("".join("-" * w for w in widths))
    for row in table:
        print(fmt(row))
    print()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: list[tuple[str, str, dict[str, Any]]],
    pairwise_rows: list[tuple[str, str, float | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        # Section 1: per-method metrics
        fieldnames = [
            "kind", "name",
            "balanced_accuracy", "cohen_kappa", "accuracy",
            "total_valid", "gold_supported", "gold_not_supported",
            "pred_supported", "pred_not_supported",
            "tp", "tn", "fp", "fn",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for kind, name, stats in rows:
            row = {"kind": kind, "name": name}
            for k in fieldnames[2:]:
                v = stats.get(k)
                if k in {"balanced_accuracy", "cohen_kappa", "accuracy"} and v is not None:
                    row[k] = f"{v * 100:.2f}%"
                else:
                    row[k] = v
            writer.writerow(row)

        # Section 2: inter-judge pairwise kappa matrix
        models_in_pairs = list(dict.fromkeys(m for ma, mb, _ in pairwise_rows for m in (ma, mb)))
        kappa_lookup: dict[tuple[str, str], float | None] = {}
        for ma, mb, k in pairwise_rows:
            kappa_lookup[(ma, mb)] = k
            kappa_lookup[(mb, ma)] = k

        handle.write("\n")
        pair_writer = csv.writer(handle)
        pair_writer.writerow([""] + models_in_pairs)
        for ma in models_in_pairs:
            row = [ma]
            for mb in models_in_pairs:
                if ma == mb:
                    row.append("—")
                else:
                    k = kappa_lookup.get((ma, mb))
                    row.append(pct(k) if k is not None else "n/a")
            pair_writer.writerow(row)
    print(f"CSV saved to: {path}")


# ---------------------------------------------------------------------------
# OW-I / ISP / Dawid-Skene (inline — no pre-step needed)
# ---------------------------------------------------------------------------

_ENCODE = {"Supported": 1, "Not Supported": 0}
_DECODE = {1: "Supported", 0: "Not Supported"}


def _extract_int_votes(
    records: list[dict[str, Any]], models: list[str]
) -> dict[str, list[int | None]]:
    vote_matrix: dict[str, list[int | None]] = {m: [] for m in models}
    for rec in records:
        outputs = rec.get("model_outputs", {})
        for m in models:
            vote_matrix[m].append(_ENCODE.get(outputs.get(m, {}).get("label")))
    return vote_matrix


def _votes_by_sample(
    models: list[str], vm: dict[str, list[int | None]]
) -> list[list[int | None]]:
    n = len(next(iter(vm.values())))
    return [[vm[m][i] for m in models] for i in range(n)]


def _mv_int(votes_per_sample: list[list[int | None]]) -> list[int | None]:
    out: list[int | None] = []
    for vs in votes_per_sample:
        valid = [v for v in vs if v is not None]
        if not valid:
            out.append(None); continue
        c = Counter(valid); top = c.most_common()
        out.append(top[0][0] if (len(top) == 1 or top[0][1] != top[1][1])
                   else next((v for v in vs if v is not None), None))
    return out


def _isp(
    models: list[str], vm: dict[str, list[int | None]],
    max_iter: int = 20, eps: float = 1e-6,
) -> list[int | None]:
    n = len(next(iter(vm.values())))
    soft: list[float] = [0.8 if v == 1 else (0.2 if v == 0 else 0.5)
                         for v in _mv_int(_votes_by_sample(models, vm))]
    for _ in range(max_iter):
        old = list(soft)
        weights: dict[str, float] = {}
        for m in models:
            num = denom = 0.0
            for i in range(n):
                v = vm[m][i]
                if v is None: continue
                p = soft[i]
                num += p if v == 1 else (1.0 - p); denom += 1.0
            acc = max(0.51, min(1.0 - eps, num / denom if denom > 0 else 0.5))
            weights[m] = math.log(acc / (1.0 - acc))
        for i in range(n):
            score = sum((weights[m] if vm[m][i] == 1 else -weights[m])
                        for m in models if vm[m][i] is not None)
            soft[i] = 1.0 / (1.0 + math.exp(-score))
        if max(abs(a - b) for a, b in zip(soft, old)) < eps:
            break
    return [1 if p > 0.5 else (0 if p < 0.5 else None) for p in soft]


def _owi(
    models: list[str], vm: dict[str, list[int | None]],
    isp_labels: list[int | None], eps: float = 1e-6,
) -> list[int | None]:
    n = len(next(iter(vm.values())))
    accs: dict[str, float] = {}
    for m in models:
        correct = total = 0
        for i in range(n):
            if vm[m][i] is not None and isp_labels[i] is not None:
                total += 1
                if vm[m][i] == isp_labels[i]: correct += 1
        accs[m] = max(0.5 + eps, min(1.0 - eps, correct / total if total > 0 else 0.5))
    w = {m: math.log(accs[m] / (1.0 - accs[m])) for m in models}
    out: list[int | None] = []
    for i in range(n):
        s1 = sum(w[m] for m in models if vm[m][i] == 1)
        s0 = sum(w[m] for m in models if vm[m][i] == 0)
        out.append(1 if s1 > s0 else (0 if s0 > s1 else None))
    return out


def _dawid_skene(
    models: list[str], vm: dict[str, list[int | None]],
    max_iter: int = 100, tol: float = 1e-6,
    init_p_y1: list[float] | None = None,
) -> tuple[list[int | None], dict[str, float], dict[str, float]]:
    n = len(next(iter(vm.values())))
    if init_p_y1 is not None:
        p_y1: list[float] = list(init_p_y1)
    else:
        p_y1 = [0.8 if v == 1 else (0.2 if v == 0 else 0.5)
                for v in _mv_int(_votes_by_sample(models, vm))]
    alpha: dict[str, float] = {m: 0.8 for m in models}
    beta:  dict[str, float] = {m: 0.8 for m in models}
    LOG_EPS = 1e-10
    for _ in range(max_iter):
        old = list(p_y1)
        for m in models:
            na = da = nb = db = 0.0
            for i in range(n):
                v = vm[m][i]
                if v is None: continue
                py1, py0 = p_y1[i], 1.0 - p_y1[i]
                da += py1; db += py0
                if v == 1: na += py1
                else:      nb += py0
            alpha[m] = max(0.5, min(0.99, na / da if da > 0 else 0.8)) # sensitivity
            beta[m]  = max(0.5, min(0.99, nb / db if db > 0 else 0.8)) # specificity
        prevalence = max(LOG_EPS, min(1.0 - LOG_EPS, sum(p_y1) / n))
        for i in range(n):
            lp1 = math.log(prevalence); lp0 = math.log(1.0 - prevalence)
            for m in models:
                v = vm[m][i]
                if v is None: continue
                if v == 1:
                    lp1 += math.log(alpha[m] + LOG_EPS)
                    lp0 += math.log(1.0 - beta[m] + LOG_EPS)
                else:
                    lp1 += math.log(1.0 - alpha[m] + LOG_EPS)
                    lp0 += math.log(beta[m] + LOG_EPS)
            lm = max(lp1, lp0)
            p1, p0 = math.exp(lp1 - lm), math.exp(lp0 - lm)
            p_y1[i] = p1 / (p1 + p0)
        if max(abs(a - b) for a, b in zip(p_y1, old)) < tol:
            break
    labels = [1 if p > 0.5 else (0 if p < 0.5 else None) for p in p_y1]
    return labels, alpha, beta


# ---------------------------------------------------------------------------
# Reliability ranking helpers
# ---------------------------------------------------------------------------

def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order, 1):
        ranks[idx] = float(rank)
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx  = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy  = sum((yi - my) ** 2 for yi in y) ** 0.5
    return num / (dx * dy) if dx * dy > 0 else 0.0


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _judge_reliability_analysis(
    models: list[str],
    vm: dict[str, list[int | None]],
    ds_alpha: dict[str, float],
    ds_beta: dict[str, float],
    true_bacc: dict[str, float],
    n_reinit: int = 20,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict:
    n = len(next(iter(vm.values())))
    rng = random.Random(seed)

    # Step A — label-free reliability score
    reliability = {m: 0.5 * (ds_alpha[m] + ds_beta[m]) for m in models}
    ranked_by_rel  = sorted(models, key=lambda m: reliability[m], reverse=True)
    ranked_by_bacc = sorted(models, key=lambda m: true_bacc.get(m, 0.0), reverse=True)

    # Step B — print ranking table + validation correlations
    rel_vals  = [reliability[m] for m in models]
    bacc_vals = [true_bacc.get(m, 0.0) for m in models]
    spearman_r = _spearman(rel_vals, bacc_vals)
    pearson_r  = _pearson(rel_vals, bacc_vals)

    print("Dawid-Skene judge reliability ranking:")
    header = f"  {'Judge':<42} {'DS Reliability':>16} {'True Bacc':>11} {'DS Rank':>9} {'Bacc Rank':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in ranked_by_rel:
        ds_r   = ranked_by_rel.index(m) + 1
        bacc_r = ranked_by_bacc.index(m) + 1
        tb     = true_bacc.get(m)
        tb_str = f"{tb:.4f}" if tb is not None else "  n/a"
        print(f"  {m:<42} {reliability[m]:>16.4f} {tb_str:>11} {ds_r:>9} {bacc_r:>10}")
    print(f"\n  Validation (N={len(models)} judges): "
          f"Spearman r = {spearman_r:.3f}   Pearson r = {pearson_r:.3f}")
    print("  (small N → treat as indicative, not definitive)\n")

    # Step C — robustness: random inits + bootstrap
    reliability_runs: list[dict[str, float]] = []

    for _ in range(n_reinit):
        init = [max(0.01, min(0.99, rng.gauss(0.5, 0.25))) for _ in range(n)]
        _, a, b = _dawid_skene(models, vm, init_p_y1=init)
        reliability_runs.append({m: 0.5 * (a[m] + b[m]) for m in models})

    indices = list(range(n))
    for _ in range(n_bootstrap):
        boot_idx = [rng.choice(indices) for _ in range(n)]
        vm_boot  = {m: [vm[m][i] for i in boot_idx] for m in models}
        _, a, b  = _dawid_skene(models, vm_boot)
        reliability_runs.append({m: 0.5 * (a[m] + b[m]) for m in models})

    rank_corrs: list[float] = []
    for i, run_a in enumerate(reliability_runs):
        for run_b in reliability_runs[i + 1:]:
            xa = [run_a[m] for m in models]
            xb = [run_b[m] for m in models]
            rank_corrs.append(_spearman(xa, xb))
    avg_rank_corr = sum(rank_corrs) / len(rank_corrs) if rank_corrs else None

    nominal_top   = ranked_by_rel[0]
    top_count     = sum(1 for run in reliability_runs if max(run, key=run.get) == nominal_top)
    top_stability = top_count / len(reliability_runs) if reliability_runs else None

    total_runs = len(reliability_runs)
    print(f"Robustness ({n_reinit} random inits + {n_bootstrap} bootstrap resamples):")
    if avg_rank_corr is not None:
        print(f"  Avg rank correlation across runs : {avg_rank_corr:.4f}")
    if top_stability is not None:
        print(f"  Top judge '{nominal_top.split('/')[-1]}' stable in : "
              f"{top_count}/{total_runs} runs ({100*top_stability:.1f}%)")
    print()

    return {
        "reliability_score":             {m: round(reliability[m], 4) for m in models},
        "ds_rank":                        {m: ranked_by_rel.index(m) + 1 for m in models},
        "true_bacc":                      {m: round(true_bacc[m], 4) if m in true_bacc else None for m in models},
        "true_bacc_rank":                 {m: ranked_by_bacc.index(m) + 1 for m in models},
        "validation_spearman_r":          round(spearman_r, 4),
        "validation_pearson_r":           round(pearson_r, 4),
        "robustness_avg_rank_corr":       round(avg_rank_corr, 4) if avg_rank_corr is not None else None,
        "robustness_top_judge":           nominal_top,
        "robustness_top_stability_pct":   round(100 * top_stability, 1) if top_stability is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MEMERAG judgement and aggregator results.")
    parser.add_argument("--lang", default="en", help="Language subfolder (en, es, de, hi, fr, ...)")
    parser.add_argument("--exclude", default=None,
                        help="Model ID to exclude from the panel (e.g. gemini-2.5-flash-lite)")
    args = parser.parse_args()


    excl_folder = ("excl_" + args.exclude.split("/")[-1].lower().replace(".", "_").replace("-", "_")) if args.exclude else None
    JUDGEMENT_JSON, AGGREGATOR_JSON, _, RANDOM_AGG_JSON = _build_paths(args.lang, excl_folder)

    if not JUDGEMENT_JSON.exists():
        print(f"[ERROR] Judgement file not found: {JUDGEMENT_JSON}")
        sys.exit(1)

    judgement_records = load_records(JUDGEMENT_JSON)
    if not judgement_records:
        print(f"[ERROR] No records found in {JUDGEMENT_JSON}")
        sys.exit(1)

    available_models = set(judgement_records[0].get("model_outputs", {}).keys())

    # Desired display order — any model not listed here falls to the end
    MODEL_ORDER = [
        "gemini-2.5-flash",
        "google/gemma-4-26b-a4b-it-maas",
        "gemini-2.5-flash-lite",
        "meta/llama-3.3-70b-instruct-maas",
    ]
    models = [m for m in MODEL_ORDER if m in available_models] + \
             [m for m in available_models if m not in MODEL_ORDER]

    if args.exclude:
        models = [m for m in models if m != args.exclude]
        print(f"[INFO] Excluding model from panel: {args.exclude}")

    print(f"Judgement file : {JUDGEMENT_JSON}")
    print(f"Aggregator file: {AGGREGATOR_JSON}")
    print(f"Records loaded : {len(judgement_records)}")
    gold_dist = Counter(r.get("gold_label") for r in judgement_records)
    print(f"Gold distribution: {dict(gold_dist)}\n")

    # Response analysis
    print_response_analysis(judgement_records)

    # Individual model metrics (in display order)
    all_metric_rows: list[tuple[str, str, dict[str, Any]]] = []
    for model in models:
        stats = compute_metrics(
            judgement_records,
            pred_getter=lambda r, m=model: r.get("model_outputs", {}).get(m, {}).get("label"),
        )
        all_metric_rows.append(("individual", model, stats))

    # Tiebreak order: best individual balanced accuracy first (data-driven, not hardcoded)
    tiebreak_models = sorted(
        models,
        key=lambda m: next(
            (s["balanced_accuracy"] for k, nm, s in all_metric_rows if nm == m and k == "individual"),
            None,
        ) or 0.0,
        reverse=True,
    )
    print(f"Tiebreak order (best bacc first): {[m.split('/')[-1] for m in tiebreak_models]}")
    print(f"Tiebreak order (random/MODEL_ORDER): {[m.split('/')[-1] for m in models]}\n")

    # Panel majority vote — recompute from current labels.
    # tiebreak_order determines which model wins a 2-2 tie.
    def _fresh_majority(r: dict[str, Any], tiebreak_order: list[str]) -> str | None:
        # Use only models in tiebreak_order so --exclude is respected
        votes = [
            r.get("model_outputs", {}).get(m, {}).get("label")
            for m in tiebreak_order
        ]
        valid = [v for v in votes if v in LABELS]
        if not valid:
            return None
        c: Counter = Counter(valid)
        top = c.most_common()
        if len(top) == 1 or top[0][1] != top[1][1]:
            return top[0][0]
        for m in tiebreak_order:
            label = r.get("model_outputs", {}).get(m, {}).get("label")
            if label in LABELS:
                return label
        return None

    # random tiebreaker = MODEL_ORDER (reproducible baseline, no gold required)
    panel_stats_random = compute_metrics(
        judgement_records,
        pred_getter=lambda r: _fresh_majority(r, models),
    )
    all_metric_rows.append(("majority_vote", "panel_majority_random", panel_stats_random))

    # best tiebreaker = highest individual bacc (upper bound — requires gold)
    panel_stats_best = compute_metrics(
        judgement_records,
        pred_getter=lambda r: _fresh_majority(r, tiebreak_models),
    )
    all_metric_rows.append(("majority_vote", "panel_majority_best", panel_stats_best))

    # Aggregator metrics
    agg_model_name = "aggregator"
    if AGGREGATOR_JSON.exists():
        agg_records = load_records(AGGREGATOR_JSON)
        agg_meta = load_meta(AGGREGATOR_JSON)
        agg_model_name = agg_meta.get("aggregator_model", "aggregator")

        print_aggregator_analysis(agg_records, agg_model_name)

        # positional match — MEMERAG has duplicate sample_ids
        merged_agg = []
        if len(agg_records) == len(judgement_records):
            for rec, agg_rec in zip(judgement_records, agg_records):
                m = dict(rec)
                m["__aggregator_label"] = agg_rec.get("aggregator_label")
                merged_agg.append(m)
        else:
            # fallback to id-based if lengths differ (e.g. partial aggregator run)
            agg_by_id = {r.get("sample_id"): r for r in agg_records}
            for rec in judgement_records:
                m = dict(rec)
                agg_rec = agg_by_id.get(rec.get("sample_id"))
                if agg_rec:
                    m["__aggregator_label"] = agg_rec.get("aggregator_label")
                merged_agg.append(m)

        agg_stats = compute_metrics(
            merged_agg,
            pred_getter=lambda r: r.get("__aggregator_label"),
        )
        all_metric_rows.append(("aggregator", agg_model_name, agg_stats))
    else:
        print(f"[INFO] Aggregator file not found, skipping: {AGGREGATOR_JSON}\n")

    # Random aggregator metrics
    if RANDOM_AGG_JSON.exists():
        rand_records = load_records(RANDOM_AGG_JSON)
        print_aggregator_analysis(rand_records, "random_aggregator")
        merged_rand = []
        if len(rand_records) == len(judgement_records):
            for rec, rand_rec in zip(judgement_records, rand_records):
                m = dict(rec)
                m["__rand_label"] = rand_rec.get("aggregator_label")
                merged_rand.append(m)
        else:
            rand_by_id = {r.get("sample_id"): r for r in rand_records}
            for rec in judgement_records:
                m = dict(rec)
                rand_rec = rand_by_id.get(rec.get("sample_id"))
                if rand_rec:
                    m["__rand_label"] = rand_rec.get("aggregator_label")
                merged_rand.append(m)
        rand_stats = compute_metrics(merged_rand, pred_getter=lambda r: r.get("__rand_label"))
        all_metric_rows.append(("aggregator", "random_aggregator", rand_stats))
    else:
        print(f"[INFO] Random aggregator file not found, skipping: {RANDOM_AGG_JSON}\n")

    # OW-I, ISP, Dawid-Skene — computed inline from the current judge panel
    vote_matrix = _extract_int_votes(judgement_records, models)
    isp_int  = _isp(models, vote_matrix)
    owi_int  = _owi(models, vote_matrix, isp_int)
    ds_int, ds_alpha, ds_beta = _dawid_skene(models, vote_matrix)
    isp_strs = [_DECODE.get(v) for v in isp_int]
    owi_strs = [_DECODE.get(v) for v in owi_int]
    ds_strs  = [_DECODE.get(v) for v in ds_int]

    for rec, ow, is_, ds in zip(judgement_records, owi_strs, isp_strs, ds_strs):
        rec["__owi_label"] = ow
        rec["__isp_label"] = is_
        rec["__ds_label"]  = ds

    owi_stats = compute_metrics(judgement_records, pred_getter=lambda r: r.get("__owi_label"))
    isp_stats = compute_metrics(judgement_records, pred_getter=lambda r: r.get("__isp_label"))
    ds_stats  = compute_metrics(judgement_records, pred_getter=lambda r: r.get("__ds_label"))
    all_metric_rows.append(("weighted_agg", "OW-I",        owi_stats))
    all_metric_rows.append(("weighted_agg", "ISP",         isp_stats))
    all_metric_rows.append(("weighted_agg", "Dawid-Skene", ds_stats))

    print_metrics_table("All metrics:", all_metric_rows)

    # Dawid-Skene reliability ranking + validation + robustness
    true_bacc_per_judge = {
        nm: s["balanced_accuracy"]
        for k, nm, s in all_metric_rows
        if k == "individual" and nm in models and s["balanced_accuracy"] is not None
    }
    reliability_result = _judge_reliability_analysis(
        models, vote_matrix, ds_alpha, ds_beta, true_bacc_per_judge
    )

    pairwise_rows = print_interjudge_table(judgement_records, models)

    out_dir = JUDGEMENT_JSON.parent / excl_folder if excl_folder else JUDGEMENT_JSON.parent
    csv_path = out_dir / f"{JUDGEMENT_JSON.stem}_metrics.csv"
    write_csv(csv_path, all_metric_rows, pairwise_rows)

    # ── All-method pairwise kappa (individual judges + MV + OW-I + ISP + DS + Agg)
    # Collect per-sample predictions for every method
    method_preds: dict[str, list[str | None]] = {}
    for model in models:
        method_preds[model] = [
            r.get("model_outputs", {}).get(model, {}).get("label") for r in judgement_records
        ]
    method_preds["MV (random)"]  = [_fresh_majority(r, models)          for r in judgement_records]
    method_preds["MV (best)"]    = [_fresh_majority(r, tiebreak_models)  for r in judgement_records]
    method_preds["OW-I"]         = owi_strs
    method_preds["ISP"]          = isp_strs
    method_preds["Dawid-Skene"]  = ds_strs

    if AGGREGATOR_JSON.exists():
        agg_records = load_records(AGGREGATOR_JSON)
        if len(agg_records) == len(judgement_records):
            agg_meta = load_meta(AGGREGATOR_JSON)
            agg_name = agg_meta.get("aggregator_model", "Aggregator")
            method_preds[f"Agg ({agg_name.split('/')[-1][:15]})"] = [
                r.get("aggregator_label") for r in agg_records
            ]

    if RANDOM_AGG_JSON.exists():
        rand_records = load_records(RANDOM_AGG_JSON)
        if len(rand_records) == len(judgement_records):
            method_preds["Agg (random)"] = [r.get("aggregator_label") for r in rand_records]

    # Compute all pairwise kappas
    all_names = list(method_preds.keys())
    method_pairwise: dict[str, float | None] = {}
    for i, name_a in enumerate(all_names):
        for name_b in all_names[i + 1:]:
            k = kappa_from_lists(method_preds[name_a], method_preds[name_b])
            method_pairwise[f"{name_a} vs {name_b}"] = round(k, 4) if k is not None else None

    import json
    kappa_json_path = out_dir / f"{JUDGEMENT_JSON.stem}_method_kappa.json"
    with kappa_json_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "method_pairwise_kappa": method_pairwise,
            "dawid_skene_judge_reliability": {
                m: {
                    "sensitivity_alpha": round(ds_alpha[m], 4),
                    "specificity_beta":  round(ds_beta[m], 4),
                }
                for m in models
            },
            "dawid_skene_reliability_analysis": reliability_result,
        }, fh, indent=2)
    print(f"Method pairwise kappa saved to: {kappa_json_path}")


if __name__ == "__main__":
    main()
