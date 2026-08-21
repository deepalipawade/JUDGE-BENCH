"""
MEMERAG aggregation pipeline.

Reads judge outputs produced by judge_memerag.py and runs:
  1. Algorithmic aggregation  (OWI, ISP, DS, majority) — no API calls
  2. LLM aggregation          (random, fixed, best_bacc, ds_rank) — API calls

Usage:
    python pipeline/pipeline.py --lang en
    python pipeline/pipeline.py --lang en --dry-run
    python pipeline/pipeline.py --lang en --no-llm-agg

"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from utils.aggregation import (
    DECODE, ENCODE, extract_vote_matrix,
    run_dawid_skene, run_isp, run_iwmv, run_majority, run_owi,
)
from utils.mace import run_mace
from utils.api import (
    SERVICE_ACCOUNT_PATH, dispatch, init_clients,
)
from utils.io import load_json, save_json
from utils.labels import extract_label_and_reason, normalize_label
from utils.metrics import (
    compute_bacc, compute_cohen_kappa, compute_full_metrics,
    fmt, kappa_from_lists, pct, print_metrics_table,
)

ROOT = Path(__file__).resolve().parents[2]

_RESULTS_ROOTS = {
    "memerag": ROOT / "results_tmp" / "memerag_ext",
    "qags":    ROOT / "results_tmp" / "qags",
}

def _results_root() -> Path:
    return _RESULTS_ROOTS.get(getattr(config, "DATASET", "memerag"), _RESULTS_ROOTS["memerag"])

RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"   # kept for any legacy direct references

AGGREGATOR_PROMPT = (
    "You are a strict factual-consistency evaluator.\n"
    "Task: Decide whether the answer segment is fully supported by the evidence passages.\n\n"
    "Use the provided judge outputs as evidence, but do not blindly follow them.\n"
    "If the judges disagree, prefer the reasoning most directly grounded in the passages.\n\n"
    "Output format:\n"
    "<Answer>Supported</Answer> or <Answer>Not Supported</Answer>\n"
    "<Reasoning>Your concise reasoning here</Reasoning>\n\n"
    "ORIGINAL TASK:\n{task_prompt}\n\n"
    "JUDGE RESPONSES:\n{judge_responses}\n"
)

TASK_PROMPT = (
    "Evidence Passages:\n{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer Segment:\n{answer_segment}"
)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx  = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy  = sum((yi - my) ** 2 for yi in y) ** 0.5
    return num / (dx * dy) if dx * dy > 0 else 0.0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order, 1):
        ranks[idx] = float(rank)
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _model_bacc(records: list[dict], model: str) -> float | None:
    flat = [
        {"gold_label": r.get("gold_label"),
         "_lbl": (r.get("model_outputs") or {}).get(model, {}).get("label")}
        for r in records
    ]
    return compute_bacc(flat, "_lbl")


def _pairwise_kappa(records: list[dict], ma: str, mb: str) -> float | None:
    a = [(r.get("model_outputs") or {}).get(ma, {}).get("label") for r in records]
    b = [(r.get("model_outputs") or {}).get(mb, {}).get("label") for r in records]
    return kappa_from_lists(a, b)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def active_proposers() -> list[str]:
    return [
        m for m in config.PROPOSERS
        if m not in config.IGNORE_MODELS and m != config.EXCLUDE_MODEL
    ]


def out_dir(lang: str) -> Path:
    base = _results_root() / lang
    if config.EXCLUDE_MODEL:
        short = config.EXCLUDE_MODEL.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        return base / f"excl_{short}"
    return base


_QAGS_LABEL_MAP = {"yes": "Supported", "no": "Not Supported"}


def _normalize_qags_records(records: list[dict]) -> list[dict]:
    """Convert QAGS yes/no labels → Supported/Not Supported in-place."""
    for rec in records:
        majority = rec.get("majority_human")
        rec["gold_label"] = _QAGS_LABEL_MAP.get(majority, majority)
        rec.setdefault("sample_id", rec.get("id"))
        for m, info in (rec.get("model_outputs") or {}).items():
            if isinstance(info, dict) and info.get("label") in _QAGS_LABEL_MAP:
                info["label"] = _QAGS_LABEL_MAP[info["label"]]
    return records


def build_prompt(record: dict[str, Any], panel: list[str]) -> str:
    if getattr(config, "DATASET", "memerag") != "memerag":
        task = record.get("prompt", "")
    else:
        context = "\n".join(f"{i+1}. {t}" for i, t in enumerate(record.get("context_texts", [])))
        task = TASK_PROMPT.format(
            context=context,
            query=record.get("query", ""),
            answer_segment=record.get("answer_segment", ""),
        )
    responses = ""
    for m in panel:
        info   = record.get("model_outputs", {}).get(m, {})
        label  = info.get("label")  or "N/A"
        reason = info.get("reason") or info.get("reasoning") or "N/A"
        responses += f"Judge {m}:\n  Label: {label}\n  Reasoning: {reason}\n\n"
    return AGGREGATOR_PROMPT.format(task_prompt=task, judge_responses=responses)


def _bacc_from_records(records: list[dict], model: str) -> float | None:
    """Compute balanced accuracy for a model directly from judge records."""
    pos_correct = pos_total = neg_correct = neg_total = 0
    for rec in records:
        gold = rec.get("gold_label")
        out  = (rec.get("model_outputs") or {}).get(model)
        label = out.get("label") if isinstance(out, dict) else None
        if gold == "Supported":
            pos_total += 1
            if label == "Supported":
                pos_correct += 1
        elif gold == "Not Supported":
            neg_total += 1
            if label == "Not Supported":
                neg_correct += 1
    if pos_total == 0 or neg_total == 0:
        return None
    return (pos_correct / pos_total + neg_correct / neg_total) / 2


def pick_best_bacc_model(records: list[dict], proposers: list[str]) -> str:
    best_model, best_bacc = None, -1.0
    for m in proposers:
        bacc = _bacc_from_records(records, m)
        if bacc is not None and bacc > best_bacc:
            best_bacc, best_model = bacc, m
    if best_model is None:
        print(f"[WARN] Could not compute bacc from records, falling back to {proposers[0]}")
        best_model = proposers[0]
    return best_model


def pick_worse_bacc_model(records: list[dict], proposers: list[str]) -> str:
    worst_model, worst_bacc = None, float("inf")
    for m in proposers:
        bacc = _bacc_from_records(records, m)
        if bacc is not None and bacc < worst_bacc:
            worst_bacc, worst_model = bacc, m
    if worst_model is None:
        print(f"[WARN] Could not compute bacc from records, falling back to {proposers[-1]}")
        worst_model = proposers[-1]
    return worst_model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(proposers: list[str]) -> None:
    if config.AGGREGATOR_LLM == "fixed":
        if config.AGGREGATOR_MODEL not in proposers:
            print(
                f"[ERROR] AGGREGATOR_MODEL '{config.AGGREGATOR_MODEL}' is not in active proposers.\n"
                f"  Active proposers: {proposers}\n"
                f"  The aggregator LLM must be one of the proposer models."
            )
            sys.exit(1)


def check_proposers(records: list[dict], proposers: list[str]) -> None:
    print(f"\nChecking proposer outputs ({len(proposers)} models, {len(records)} samples):")
    missing_models = []
    for m in proposers:
        valid = sum(
            1 for r in records
            if normalize_label((r.get("model_outputs") or {}).get(m, {}).get("label")) is not None
        )
        if valid == 0:
            print(f"  [MISSING] {m}: no outputs found")
            missing_models.append(m)
        elif valid < len(records):
            print(f"  [PARTIAL] {m}: {valid}/{len(records)} valid labels")
        else:
            print(f"  [OK]      {m}: {valid}/{len(records)} valid labels")
    if missing_models:
        print(f"\n[ERROR] Missing judge outputs for: {missing_models}")
        print(f"  Run: python judge_memerag.py --lang <lang> --all")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Inter-judge pairwise kappa
# ---------------------------------------------------------------------------

def print_interjudge_kappa(
    records: list[dict], models: list[str]
) -> list[tuple[str, str, float | None]]:
    """Print NxN pairwise kappa matrix between judge models. Returns pair list."""
    kappa_matrix: dict[tuple[str, str], float | None] = {}
    pairs: list[tuple[str, str, float | None]] = []
    for ma, mb in combinations(models, 2):
        k = _pairwise_kappa(records, ma, mb)
        kappa_matrix[(ma, mb)] = k
        kappa_matrix[(mb, ma)] = k
        pairs.append((ma, mb, k))

    short = {m: m.split("/")[-1][:18] for m in models}
    col_w = max(len(s) for s in short.values()) + 2

    print("Inter-judge pairwise Cohen's Kappa:")
    header_pad = 44
    print(f"  {'':>{header_pad}}" + "".join(short[m].ljust(col_w) for m in models))
    print("  " + "-" * (header_pad + col_w * len(models)))
    for ma in models:
        row = f"  {short[ma]:<{header_pad}}"
        for mb in models:
            if ma == mb:
                cell = "—"
            else:
                k = kappa_matrix.get((ma, mb))
                cell = f"{pct(k):.2f}" if k is not None else "n/a"
            row += cell.ljust(col_w)
        print(row)
    print()
    return pairs


# ---------------------------------------------------------------------------
# DS reliability analysis (ranking + validation + robustness)
# ---------------------------------------------------------------------------

def _ds_reliability_analysis(
    models: list[str],
    matrix: dict[str, list[int | None]],
    ds_alpha: dict[str, float],
    ds_beta: dict[str, float],
    records: list[dict],
    n_reinit: int = 20,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    n = len(next(iter(matrix.values())))
    rng = random.Random(seed)

    reliability = {m: 0.5 * (ds_alpha[m] + ds_beta[m]) for m in models}
    ranked_by_rel  = sorted(models, key=lambda m: reliability[m], reverse=True)
    true_bacc      = {m: _model_bacc(records, m) for m in models}
    ranked_by_bacc = sorted(models, key=lambda m: true_bacc.get(m) or 0.0, reverse=True)

    rel_vals   = [reliability[m] for m in models]
    bacc_vals  = [true_bacc.get(m) or 0.0 for m in models]
    spearman_r = _spearman(rel_vals, bacc_vals)
    pearson_r  = _pearson(rel_vals, bacc_vals)

    col_w = max(len(m) for m in models) + 2
    print("Dawid-Skene judge reliability ranking:")
    print(f"  {'Judge':<{col_w}} {'DS Rel':>10} {'True Bacc':>11} {'DS Rank':>9} {'Bacc Rank':>10}")
    print("  " + "-" * (col_w + 44))
    for m in ranked_by_rel:
        ds_r   = ranked_by_rel.index(m) + 1
        bacc_r = ranked_by_bacc.index(m) + 1
        print(f"  {m:<{col_w}} {reliability[m]:>10.4f} {fmt(true_bacc.get(m)):>11} {ds_r:>9} {bacc_r:>10}")
    print(f"\n  Spearman r = {spearman_r:.3f}   Pearson r = {pearson_r:.3f}")
    print("  (small N → treat as indicative)\n")

    # Robustness: random inits + bootstrap resamples
    reliability_runs: list[dict[str, float]] = []
    for _ in range(n_reinit):
        init = [max(0.01, min(0.99, rng.gauss(0.5, 0.25))) for _ in range(n)]
        _, a, b, _ = run_dawid_skene(models, matrix, init_p_y1=init)
        reliability_runs.append({m: 0.5 * (a[m] + b[m]) for m in models})

    indices = list(range(n))
    for _ in range(n_bootstrap):
        boot_idx = [rng.choice(indices) for _ in range(n)]
        vm_boot  = {m: [matrix[m][i] for i in boot_idx] for m in models}
        _, a, b, _ = run_dawid_skene(models, vm_boot)
        reliability_runs.append({m: 0.5 * (a[m] + b[m]) for m in models})

    rank_corrs = [
        _spearman([ra[m] for m in models], [rb[m] for m in models])
        for i, ra in enumerate(reliability_runs)
        for rb in reliability_runs[i + 1:]
    ]
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
              f"{top_count}/{total_runs} runs ({100 * top_stability:.1f}%)")
    print()

    return {
        "reliability_score":            {m: round(reliability[m], 4) for m in models},
        "ds_rank":                      {m: ranked_by_rel.index(m) + 1 for m in models},
        "sensitivity_alpha":            {m: round(ds_alpha[m], 4) for m in models},
        "specificity_beta":             {m: round(ds_beta[m], 4) for m in models},
        "true_bacc":                    {m: pct(true_bacc.get(m)) for m in models},
        "true_bacc_rank":               {m: ranked_by_bacc.index(m) + 1 for m in models},
        "validation_spearman_r":        round(spearman_r, 4),
        "validation_pearson_r":         round(pearson_r, 4),
        "robustness_avg_rank_corr":     round(avg_rank_corr, 4) if avg_rank_corr is not None else None,
        "robustness_top_judge":         nominal_top,
        "robustness_top_stability_pct": round(100 * top_stability, 1) if top_stability is not None else None,
    }


# ---------------------------------------------------------------------------
# Vote matrix CSV
# ---------------------------------------------------------------------------

def save_vote_matrix(
    records: list[dict],
    active_models: list[str],
    matrix: dict[str, list[int | None]],
    lang: str,
) -> None:
    """Save raw judge votes to results_tmp/votes/votes_{lang}.csv.

    Columns: sample_id, gold_label, <model1>, <model2>, ...
    Values:  1 = Supported, 0 = Not Supported, empty = missing
    """
    if getattr(config, "DATASET", "memerag") == "memerag":
        votes_dir = _results_root() / "votes"
    else:
        votes_dir = out_dir(lang)
    votes_dir.mkdir(parents=True, exist_ok=True)
    out_path = votes_dir / f"votes_{lang}.csv"

    fieldnames = ["sample_id", "gold_label"] + active_models
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, rec in enumerate(records):
            gold = ENCODE.get(rec.get("gold_label"))
            row: dict[str, Any] = {
                "sample_id":  rec.get("sample_id", ""),
                "gold_label": "" if gold is None else gold,
            }
            for m in active_models:
                lbl = matrix[m][i]
                row[m] = "" if lbl is None else lbl
            writer.writerow(row)

    print(f"  Vote matrix  → {out_path.name}  ({len(records)} samples × {len(active_models)} models)")


# ---------------------------------------------------------------------------
# Step 1: Algorithmic aggregation
# ---------------------------------------------------------------------------

def run_algo_aggregation(
    records: list[dict],
    lang: str,
    proposers: list[str],
) -> tuple[list[tuple[str, float | None, float | None]], dict | None]:
    active_models, matrix = extract_vote_matrix(records, proposers)
    save_vote_matrix(records, active_models, matrix, lang)
    table_rows: list[tuple[str, float | None, float | None]] = []
    metrics:      dict[str, Any] = {}
    method_params: dict[str, Any] = {}

    ds_alpha: dict[str, float] | None = None
    ds_beta:  dict[str, float] | None = None
    ds_reliability_result: dict | None = None

    for algo in config.AGGREGATION_ALGOS:
        print(f"  {algo.upper():<12}", end=" ", flush=True)

        if algo == "majority":
            int_labels = run_majority(active_models, matrix)

        elif algo == "isp":
            int_labels, isp_cond = run_isp(active_models, matrix)
            # Summarise cond-prob table: per model, mean P(i=1|j=0) and P(i=1|j=1) over all j≠i
            isp_p_given_0 = {
                m: sum(isp_cond.get((m, j, 1, 0), 0.5) for j in active_models if j != m)
                   / (len(active_models) - 1)
                for m in active_models
            }
            isp_p_given_1 = {
                m: sum(isp_cond.get((m, j, 1, 1), 0.5) for j in active_models if j != m)
                   / (len(active_models) - 1)
                for m in active_models
            }
            isp_gap = {m: round(isp_p_given_1[m] - isp_p_given_0[m], 4) for m in active_models}
            method_params["isp"] = {
                "mean_P_i1_given_j0": {m: round(isp_p_given_0[m], 4) for m in active_models},
                "mean_P_i1_given_j1": {m: round(isp_p_given_1[m], 4) for m in active_models},
                "gap_P1_minus_P0":    isp_gap,
            }

        elif algo == "iwmv":
            int_labels, iwmv_acc, iwmv_weights, iwmv_conv = run_iwmv(active_models, matrix)
            method_params["iwmv"] = {
                "per_model_accuracy": {m: round(iwmv_acc[m], 4) for m in active_models},
                "per_model_weight":   {m: round(iwmv_weights[m], 4) for m in active_models},
                "convergence":        iwmv_conv,
            }

        elif algo == "owi":
            int_labels, owi_acc, owi_weights = run_owi(active_models, matrix)
            method_params["owi"] = {
                "per_model_accuracy": {m: round(owi_acc[m], 4) for m in active_models},
                "per_model_weight":   {m: round(owi_weights[m], 4) for m in active_models},
            }

        elif algo == "ds":
            int_labels, ds_alpha, ds_beta, _ = run_dawid_skene(active_models, matrix)
            method_params["ds"] = {
                "per_model_sensitivity_alpha": {m: round(ds_alpha[m], 4) for m in active_models},
                "per_model_specificity_beta":  {m: round(ds_beta[m], 4) for m in active_models},
            }

        elif algo == "mace":
            int_labels, mace_competence, mace_entropy = run_mace(
                active_models, matrix, random_state=config.SEED
            )
            method_params["mace"] = {
                "per_model_competence": {m: round(mace_competence[m], 4) for m in active_models},
                "method": "vb",
                "n_restarts": 10,
                "n_iter": 50,
            }
            for r, ent in zip(records, mace_entropy):
                r["__mace_entropy"] = ent if not math.isnan(ent) else None

        else:
            print("unknown — skipping")
            continue

        key = f"__{algo}"
        for r, lbl in zip(records, [DECODE.get(l) for l in int_labels]):
            r[key] = lbl

        bacc  = compute_bacc(records, key)
        kappa = compute_cohen_kappa(records, key)
        table_rows.append((algo, bacc, kappa))
        metrics[algo] = {"balanced_accuracy": pct(bacc), "cohen_kappa": pct(kappa)}
        print(f"bacc={fmt(bacc)}  kappa={fmt(kappa)}")

        # Per-model diagnostics
        if algo == "majority":
            n_items = len(next(iter(matrix.values())))
            split_counts: dict[str, int] = {}
            for i in range(n_items):
                votes = [matrix[m][i] for m in active_models if matrix[m][i] is not None]
                n1 = sum(votes); n0 = len(votes) - n1
                key = f"{max(n1, n0)}-{min(n1, n0)}"
                split_counts[key] = split_counts.get(key, 0) + 1
            method_params["majority"] = {"vote_split_distribution": split_counts}
            print(f"    Vote split distribution ({n_items} items):")
            for split in sorted(split_counts, reverse=True):
                cnt = split_counts[split]
                bar = "█" * min(cnt, 40)
                print(f"      {split}: {cnt:>4}  {bar}")

        elif algo == "isp":
            def _isp_behavior(p0: float, gap: float) -> str:
                if p0 > 0.40:
                    return "positive outlier"   # says Supported even when all others disagree
                if p0 < 0.20 and gap > 0.65:
                    return "crowd-follower"     # calibrated, high consensus-sensitivity
                if gap < 0.50:
                    return "low-signal"         # vote barely shifts with consensus
                return "mild positive lean"     # slightly biased toward Supported

            print(f"    Cond-prob summary (averaged over all j≠i per model) — ISP:")
            print(f"      P(i=1|j=0): how often judge i says Supported when all others say Not Supported.")
            print(f"      P(i=1|j=1): how often judge i says Supported when all others agree on Supported.")
            print(f"      Gap = P(1) - P(0): larger gap → vote is more conditionally informative.")
            print()
            print(f"      {'Model':<44} {'P(i=1|j=0)':>11} {'P(i=1|j=1)':>11} {'Gap':>7}  Behavior")
            for m in active_models:
                beh = _isp_behavior(isp_p_given_0[m], isp_gap[m])
                print(f"      {m:<44} {isp_p_given_0[m]:>11.4f} {isp_p_given_1[m]:>11.4f} {isp_gap[m]:>7.4f}  {beh}")

        elif algo == "iwmv":
            cv = iwmv_conv
            conv_str = f"converged in {cv['n_iters']} iters" if cv['converged'] else f"hit max_iter={cv['n_iters']}"
            print(f"    Per-model accuracy / log-weight (IWMV)  [{conv_str}, weight spread={cv['weight_range']:.4f}]:")
            print(f"      Most reweighted: {cv['most_changed'].split('/')[-1]} (Δw={cv['max_weight_delta']:.4f})")
            print()
            for m in active_models:
                print(f"      {m:<44} acc={iwmv_acc[m]:.4f}  w={iwmv_weights[m]:.4f}")

        elif algo == "owi":
            print(f"    Per-model accuracy / log-weight (OWI):")
            for m in active_models:
                print(f"      {m:<44} acc={owi_acc[m]:.4f}  w={owi_weights[m]:.4f}")

        elif algo == "ds":
            ds_rel = {m: 0.5 * (ds_alpha[m] + ds_beta[m]) for m in active_models}
            ranked_ds = sorted(active_models, key=lambda m: ds_rel[m], reverse=True)
            print(f"    Per-model α / β / reliability (DS)  [reliability = (α+β)/2, higher = more trustworthy]:")
            print()
            print(f"      {'Model':<44} {'α(sens)':>9} {'β(spec)':>9} {'rel':>7}  rank")
            for rank, m in enumerate(ranked_ds, 1):
                floor = "  [floor hit]" if ds_alpha[m] < 0.505 or ds_beta[m] < 0.505 else ""
                print(f"      {m:<44} {ds_alpha[m]:>9.4f} {ds_beta[m]:>9.4f} {ds_rel[m]:>7.4f}  {rank:<4}{floor}")
            method_params["ds"]["per_model_reliability"] = {m: round(ds_rel[m], 4) for m in active_models}

        elif algo == "mace":
            method_params["mace"]["balanced_accuracy"] = pct(bacc)
            method_params["mace"]["cohen_kappa"]       = pct(kappa)
            ranked_mace = sorted(active_models, key=lambda m: mace_competence[m], reverse=True)
            print(f"    Per-model competence (MACE)  [higher = more reliable, lower = more spammy]:")
            print()
            print(f"      {'Model':<44} {'competence':>11}  rank")
            for rank, m in enumerate(ranked_mace, 1):
                flag = "  [?spammer]" if mace_competence[m] < 0.30 else ""
                print(f"      {m:<44} {mace_competence[m]:>11.4f}  {rank:<4}{flag}")

    # DS reliability analysis (ranking + robustness) — only if DS was run
    if ds_alpha is not None:
        print(f"\n{'='*60}\nDS RELIABILITY ANALYSIS\n{'='*60}")
        ds_reliability_result = _ds_reliability_analysis(
            active_models, matrix, ds_alpha, ds_beta, records, seed=config.SEED
        )

    out_path = out_dir(lang) / f"algo_agg_{lang}.json"
    save_json(out_path, {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "lang":                 lang,
        "methods":              config.AGGREGATION_ALGOS,
        "excluded_model":       config.EXCLUDE_MODEL,
        "models":               active_models,
        "method_params":        method_params,
        "ds_reliability_analysis": ds_reliability_result,
        "records":              records,
        "metrics":              metrics,
    })
    print(f"  → {out_path.name}")

    return table_rows, ds_reliability_result


# ---------------------------------------------------------------------------
# Step 2: LLM aggregation
# ---------------------------------------------------------------------------

def run_llm_aggregation(
    records: list[dict],
    lang: str,
    proposers: list[str],
    clients: dict,
    genai: Any,
    HttpOptions: Any,
) -> tuple[str, float | None, float | None, dict[str, str | None]] | None:
    method = config.AGGREGATOR_LLM

    # Determine fixed aggregator model
    fixed_model: str | None = None
    if method == "best_bacc":
        fixed_model = pick_best_bacc_model(records, proposers)
        print(f"  best_bacc selected: {fixed_model}")
    elif method == "worse_bacc":
        fixed_model = pick_worse_bacc_model(records, proposers)
        print(f"  worse_bacc selected: {fixed_model}")
    elif method == "fixed":
        fixed_model = config.AGGREGATOR_MODEL
    elif method == "ds_rank":
        # Read DS ranking from the already-saved algo_agg JSON (computed in Step 1)
        algo_path = out_dir(lang) / f"algo_agg_{lang}.json"
        algo_data = load_json(algo_path) if algo_path.exists() else {}
        rel_analysis = algo_data.get("ds_reliability_analysis") or {}
        ds_ranks = rel_analysis.get("ds_rank") or {}
        eligible_for_ds = {m: ds_ranks[m] for m in proposers if m in ds_ranks}
        if not eligible_for_ds:
            print(f"  [ERROR] ds_rank: no DS reliability data found in {algo_path.name}")
            print(f"          Run pipeline with 'ds' in AGGREGATION_ALGOS first.")
            sys.exit(1)
        fixed_model = min(eligible_for_ds, key=eligible_for_ds.get)
        rel_score = rel_analysis.get("reliability_score", {}).get(fixed_model)
        print(f"  ds_rank selected: {fixed_model} (rank=1, reliability={rel_score})")

    # Output path
    if method == "random":
        out_path = out_dir(lang) / f"llm_agg_random_seed{config.SEED}_{lang}.json"
    else:
        short = fixed_model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        out_path = out_dir(lang) / f"llm_agg_{method}_{short}_{lang}.json"

    # Resume — only skip samples that already have a valid label; retry errors
    completed: list[dict] = []
    if out_path.exists():
        try:
            all_saved = [r for r in load_json(out_path).get("records", []) if isinstance(r, dict)]
            for r in all_saved:
                if "__aggregator" not in r and "__llm_agg" in r:
                    r["__aggregator"] = r["__llm_agg"]
            completed = [r for r in all_saved if r.get("aggregator_label") is not None]
        except Exception:
            pass
    done_ids  = {str(r["sample_id"]) for r in completed}
    remaining = [r for r in records if str(r.get("sample_id")) not in done_ids]
    print(f"  Resume: {len(done_ids)} done, {len(remaining)} remaining (errors will be retried)")

    def checkpoint() -> None:
        save_json(out_path, {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "lang":            lang,
            "method":          method,
            "aggregator_model": fixed_model,
            "excluded_model":  config.EXCLUDE_MODEL,
            "n_samples":       len(completed),
            "records":         completed,
        })

    try:
        for idx, record in enumerate(remaining, start=len(done_ids) + 1):
            sid = record.get("sample_id")

            eligible = [
                m for m in proposers
                if isinstance((record.get("model_outputs") or {}).get(m), dict)
                and normalize_label(record["model_outputs"][m].get("label")) is not None
            ]
            if len(eligible) < 2:
                print(f"  [{idx}/{len(records)}] sample_id={sid} — skip: <2 eligible models")
                result = dict(record)
                result.update({
                    "aggregator_model": None, "aggregator_prompt": None,
                    "aggregator_label": None, "aggregator_reason": None,
                    "aggregator_raw": None, "aggregator_error": "insufficient models",
                    "aggregator_correct_gold": None, "__aggregator": None,
                })
                completed.append(result)
                continue

            if method == "random":
                agg_model = random.choice(eligible)
            else:
                agg_model = fixed_model if fixed_model in eligible else random.choice(eligible)

            print(f"  [{idx}/{len(records)}] sample_id={sid}  agg={agg_model}", end=" ... ", flush=True)
            prompt = build_prompt(record, eligible)
            raw, error = dispatch(agg_model, prompt, clients, genai, HttpOptions)
            agg_label, agg_reason = extract_label_and_reason(raw) if raw else (None, None)
            gold = record.get("gold_label")
            if error:
                print(f"→ None (err: {error})")
            else:
                print(f"→ {agg_label} (ok)")

            result = dict(record)
            result.update({
                "aggregator_model":       agg_model,
                "aggregator_prompt":      prompt,
                "aggregator_label":       agg_label,
                "aggregator_reason":      agg_reason,
                "aggregator_raw":         raw,
                "aggregator_error":       error,
                "aggregator_correct_gold": (
                    agg_label == gold if agg_label in {"Supported", "Not Supported"} else None
                ),
                "__aggregator": agg_label,
            })
            completed.append(result)

            if idx % config.CHECKPOINT_EVERY == 0 or idx == len(records):
                checkpoint()
                print(f"  [checkpoint] {len(completed)}/{len(records)} saved")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] saving checkpoint...")
        checkpoint()
        return None

    bacc  = compute_bacc(completed, "__aggregator")
    kappa = compute_cohen_kappa(completed, "__aggregator")

    agg_counts = Counter(r.get("aggregator_model") for r in completed if r.get("aggregator_model"))

    per_model_metrics: dict[str, Any] = {}
    for m in sorted(agg_counts):
        model_records = [r for r in completed if r.get("aggregator_model") == m]
        per_model_metrics[m] = {
            "balanced_accuracy": pct(compute_bacc(model_records, "__aggregator")),
            "cohen_kappa":       pct(compute_cohen_kappa(model_records, "__aggregator")),
            "n_samples":         len(model_records),
        }

    metrics = {
        "overall":   {"balanced_accuracy": pct(bacc), "cohen_kappa": pct(kappa), "n_samples": len(completed)},
        "per_model": per_model_metrics,
    }

    save_json(out_path, {
        "timestamp":                datetime.now(timezone.utc).isoformat(),
        "lang":                     lang,
        "method":                   method,
        "aggregator_model":         fixed_model,
        "excluded_model":           config.EXCLUDE_MODEL,
        "n_samples":                len(completed),
        "aggregator_selection_counts": dict(agg_counts),
        "records":                  completed,
        "metrics":                  metrics,
    })

    label_dist = Counter(r.get("aggregator_label") for r in completed)
    print(f"\n  Label distribution: {dict(label_dist)}")

    if method == "random":
        print(f"  Aggregator selection counts:")
        for m, cnt in sorted(agg_counts.items(), key=lambda x: -x[1]):
            print(f"    {m}: {cnt}")

    print(f"  bacc={fmt(bacc)}  kappa={fmt(kappa)}")
    print(f"\n  Saved → {out_path.name}")

    errors  = [r for r in completed if r.get("aggregator_error") and r.get("aggregator_error") != "insufficient models"]
    missing = [r for r in completed if r.get("aggregator_label") is None and not r.get("aggregator_error")]
    insuff  = [r for r in completed if r.get("aggregator_error") == "insufficient models"]

    excl_flag = f" --exclude {config.EXCLUDE_MODEL}" if config.EXCLUDE_MODEL else ""
    retry_cmd = f"python pipeline/pipeline.py --lang {lang}{excl_flag}"
    retry_hint: str | None = None

    if errors:
        print(f"\n  [ERRORS] {len(errors)} records with API errors:")
        for r in errors:
            print(f"    sample_id={r['sample_id']}  model={r.get('aggregator_model')}  {r.get('aggregator_error')}")
        retry_hint = retry_cmd

    if missing:
        miss_ids = " ".join(str(r["sample_id"]) for r in missing)
        print(f"\n  [MISSING] {len(missing)} records with no label and no error:")
        print(f"    sample_ids: {miss_ids}")
        retry_hint = retry_cmd

    if insuff:
        insuff_ids = " ".join(str(r["sample_id"]) for r in insuff)
        print(f"\n  [SKIPPED] {len(insuff)} records had <2 eligible models:")
        print(f"    sample_ids: {insuff_ids}")

    if not errors and not missing:
        print("\n  [OK] All records have labels.")

    row_label = "llm_random" if method == "random" else f"llm_{method}({fixed_model.split('/')[-1]})"

    # Build sample_id → aggregator_label map for method kappa computation
    sid_to_label: dict[str, str | None] = {
        str(r.get("sample_id")): r.get("__aggregator") for r in completed
    }
    return row_label, bacc, kappa, sid_to_label, retry_hint


# ---------------------------------------------------------------------------
# Post-run: method pairwise kappa + CSV
# ---------------------------------------------------------------------------

def save_method_kappa_json(
    records: list[dict],
    proposers: list[str],
    algo_keys: list[str],
    lang: str,
    llm_preds: dict[str, str | None] | None = None,
    llm_label: str = "llm_agg",
    ds_reliability_result: dict | None = None,
) -> None:
    """Pairwise kappa between all methods + DS reliability — matches old format."""
    # Display names matching old evaluate_memerag_results.py
    ALGO_DISPLAY = {"majority": "MV", "owi": "OW-I", "isp": "ISP", "iwmv": "IWMV", "ds": "Dawid-Skene"}

    method_preds: dict[str, list[str | None]] = {}
    for m in proposers:
        method_preds[m] = [(r.get("model_outputs") or {}).get(m, {}).get("label") for r in records]
    for key in algo_keys:
        method_preds[ALGO_DISPLAY.get(key, key)] = [r.get(f"__{key}") for r in records]
    if llm_preds is not None:
        method_preds[llm_label] = [llm_preds.get(str(r.get("sample_id"))) for r in records]

    all_names = list(method_preds.keys())
    pairwise: dict[str, float | None] = {}
    for na, nb in combinations(all_names, 2):
        k = kappa_from_lists(method_preds[na], method_preds[nb])
        pairwise[f"{na} vs {nb}"] = round(k, 4) if k is not None else None

    json_path = out_dir(lang) / f"method_kappa_{lang}.json"

    # Merge with existing — kappa pairs from previous LLM runs are kept
    existing: dict = {}
    if json_path.exists():
        try:
            existing = load_json(json_path)
        except Exception:
            pass
    existing_pairwise = existing.get("method_pairwise_kappa", {})
    # Drop stale entries for the same method type (e.g. old llm_best_bacc(gemma) when
    # current run is llm_best_bacc(qwen)) so only the latest run for each method is kept.
    if llm_label and "(" in llm_label:
        method_prefix = llm_label.split("(")[0]
        existing_pairwise = {
            k: v for k, v in existing_pairwise.items()
            if not any(part.split("(")[0] == method_prefix for part in k.split(" vs "))
        }
    merged_pairwise = {**existing_pairwise, **pairwise}

    output: dict[str, Any] = {"method_pairwise_kappa": merged_pairwise}

    if ds_reliability_result:
        sens  = ds_reliability_result.get("sensitivity_alpha", {})
        spec  = ds_reliability_result.get("specificity_beta", {})
        if sens:
            output["dawid_skene_judge_reliability"] = {
                m: {"sensitivity_alpha": sens[m], "specificity_beta": spec.get(m)}
                for m in sens
            }
        output["dawid_skene_reliability_analysis"] = ds_reliability_result

    save_json(json_path, output)
    print(f"  Method kappa → {json_path.name}  ({len(merged_pairwise)} pairs)")


def save_metrics_csv(
    records: list[dict],
    proposers: list[str],
    algo_keys: list[str],
    lang: str,
    llm_label: str | None = None,
    llm_sid_to_label: dict[str, str | None] | None = None,
    ds_reliability_result: dict | None = None,
) -> None:
    """Write metrics CSV.

    Section 1: per-method rows (individual, majority_vote, weighted_agg, aggregator)
               with full confusion matrix columns. LLM aggregator rows accumulate
               across runs; algo + individual rows always refresh.
    Section 2: inter-judge pairwise kappa matrix.
    Section 3: DS reliability scores (ds_rank, alpha, beta, true_bacc) — our addition.
    """
    FIELDS = ["kind", "name", "balanced_accuracy", "cohen_kappa", "gap"]
    ALGO_KIND    = {"majority": "majority_vote", "owi": "weighted_agg",
                    "isp": "weighted_agg",       "iwmv": "weighted_agg",
                    "ds": "weighted_agg",         "mace": "weighted_agg"}
    ALGO_DISPLAY = {"majority": "majority", "owi": "OW-I", "isp": "ISP", "iwmv": "IWMV",
                    "ds": "Dawid-Skene", "mace": "MACE"}

    def _pct_str(v: float | None) -> str:
        return f"{v * 100:.2f}%" if v is not None else "n/a"

    def _gap(bacc: float | None, kappa: float | None) -> str:
        if bacc is None or kappa is None:
            return "n/a"
        return _pct_str(bacc - kappa)

    def _row(kind: str, name: str, stats: dict) -> dict:
        bacc  = stats["balanced_accuracy"]
        kappa = stats["cohen_kappa"]
        return {
            "kind":               kind,
            "name":               name,
            "balanced_accuracy":  _pct_str(bacc),
            "cohen_kappa":        _pct_str(kappa),
            "gap":                _gap(bacc, kappa),
        }

    # --- Build fresh rows ---
    fresh_rows: dict[tuple[str, str], dict] = {}

    for m in proposers:
        flat = [
            {"gold_label": r.get("gold_label"),
             "__p": (r.get("model_outputs") or {}).get(m, {}).get("label")}
            for r in records
        ]
        fresh_rows[("individual", m)] = _row("individual", m, compute_full_metrics(flat, "__p"))

    for algo in algo_keys:
        kind = ALGO_KIND.get(algo, "weighted_agg")
        name = ALGO_DISPLAY.get(algo, algo)
        fresh_rows[(kind, name)] = _row(kind, name, compute_full_metrics(records, f"__{algo}"))

    if llm_label and llm_sid_to_label is not None:
        flat = [
            {"gold_label": r.get("gold_label"),
             "__p": llm_sid_to_label.get(str(r.get("sample_id")))}
            for r in records
        ]
        fresh_rows[("aggregator", llm_label)] = _row(
            "aggregator", llm_label, compute_full_metrics(flat, "__p")
        )

    # --- Load existing aggregator rows from previous LLM runs ---
    csv_path = out_dir(lang) / f"metrics_summary_{lang}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    method_prefix = llm_label.split("(")[0] if (llm_label and "(" in llm_label) else None
    existing_agg: dict[tuple[str, str], dict] = {}
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if "kind" in (reader.fieldnames or []):
                    for row in reader:
                        if row.get("kind") == "aggregator":
                            key = ("aggregator", row["name"])
                            # Skip stale entries for the same method type as current run
                            if method_prefix and row["name"].startswith(method_prefix + "("):
                                continue
                            if key not in fresh_rows:
                                bacc_str  = row.get("balanced_accuracy", "n/a")
                                kappa_str = row.get("cohen_kappa", "n/a")
                                try:
                                    gap_val = float(bacc_str.rstrip("%")) - float(kappa_str.rstrip("%"))
                                    gap_str = f"{gap_val:.2f}%"
                                except (ValueError, AttributeError):
                                    gap_str = row.get("gap", "n/a")
                                existing_agg[key] = {
                                    "kind": row["kind"], "name": row["name"],
                                    "balanced_accuracy": bacc_str,
                                    "cohen_kappa":       kappa_str,
                                    "gap":               gap_str,
                                }
        except Exception:
            pass

    all_rows = {**existing_agg, **fresh_rows}

    # --- Ordered row keys ---
    order: list[tuple[str, str]] = (
        [("individual", m) for m in proposers]
        + ([("majority_vote", "majority")] if "majority" in algo_keys else [])
        + [(ALGO_KIND[a], ALGO_DISPLAY[a]) for a in ["owi", "isp", "iwmv", "ds", "mace"] if a in algo_keys]
        + ([("aggregator", llm_label)] if llm_label else [])
        + list(existing_agg.keys())
    )
    seen: set = set()
    ordered_keys: list[tuple[str, str]] = []
    for k in order:
        if k not in seen:
            seen.add(k)
            ordered_keys.append(k)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        # Section 1: metrics table
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for key in ordered_keys:
            if key in all_rows:
                writer.writerow(all_rows[key])

        # Section 2: inter-judge pairwise Cohen's Kappa
        fh.write("\n")
        pair_writer = csv.writer(fh)
        pair_writer.writerow(["Inter-judge pairwise Cohen's Kappa"])
        pair_writer.writerow([""] + proposers)
        kappa_cache: dict[tuple[str, str], float | None] = {}
        for ma, mb in combinations(proposers, 2):
            k = _pairwise_kappa(records, ma, mb)
            kappa_cache[(ma, mb)] = k
            kappa_cache[(mb, ma)] = k
        for ma in proposers:
            row_vals: list[Any] = [ma]
            for mb in proposers:
                if ma == mb:
                    row_vals.append("—")
                else:
                    k = kappa_cache.get((ma, mb))
                    row_vals.append(f"{k * 100:.2f}%" if k is not None else "n/a")
            pair_writer.writerow(row_vals)

        # Section 3: DS reliability (our addition)
        if ds_reliability_result:
            rel   = ds_reliability_result.get("reliability_score", {})
            ranks = ds_reliability_result.get("ds_rank", {})
            tb    = ds_reliability_result.get("true_bacc", {})
            fh.write("\n")
            rel_writer = csv.DictWriter(
                fh, fieldnames=["model", "ds_rank", "ds_reliability_score", "true_bacc"]
            )
            rel_writer.writerow({"model": "Dawid-Skene Judge Reliability"})
            rel_writer.writeheader()
            for m in sorted(rel, key=lambda m: ranks.get(m, 999)):
                rel_writer.writerow({
                    "model":                m,
                    "ds_rank":              ranks.get(m),
                    "ds_reliability_score": rel.get(m),
                    "true_bacc":            tb.get(m, "n/a"),
                })

    n_agg = sum(1 for k in all_rows if k[0] == "aggregator")
    print(f"  Metrics CSV  → {csv_path.name}  ({n_agg} aggregator row(s))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Pre-parse --dataset before building the full parser so --lang choices are correct.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--dataset", default=getattr(config, "DATASET", "memerag"))
    _pre_args, _ = _pre.parse_known_args()
    dataset = _pre_args.dataset
    config.DATASET = dataset  # propagate CLI override into config module

    _LANG_CHOICES = {
        "memerag": ["en", "es", "de", "fr", "hi"],
        "qags":    ["cnndm", "xsum"],
    }
    lang_choices = _LANG_CHOICES.get(dataset, ["en", "es", "de", "fr", "hi"])

    parser = argparse.ArgumentParser(description="Aggregation pipeline (memerag or qags).")
    parser.add_argument("--dataset", default=dataset,
                        choices=list(_LANG_CHOICES.keys()),
                        help="Dataset to run (default: from config.py).")
    parser.add_argument(
        "--lang",
        required=True,
        choices=lang_choices,
        default=config.LANG,
        help="Language (memerag) or subset (qags: cnndm|xsum).",
    )
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit to first N records (useful for quick checks).")
    parser.add_argument("--no-llm-agg", action="store_true",
                        help="Skip LLM aggregation step (overrides config.AGGREGATOR_LLM).")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run, no execution.")
    args = parser.parse_args()

    lang      = args.lang
    proposers = active_proposers()

    # CLI overrides
    if args.no_llm_agg:
        config.AGGREGATOR_LLM = None

    print(f"Dataset   : {dataset}")
    print(f"Lang      : {lang}")
    print(f"Proposers : {proposers}")
    print(f"Exclude   : {config.EXCLUDE_MODEL or 'none'}")
    print(f"Algo algos: {config.AGGREGATION_ALGOS or 'none'}")
    print(f"LLM agg   : {config.AGGREGATOR_LLM or 'none'}")

    validate_config(proposers)

    if dataset == "memerag":
        input_path = _results_root() / lang / f"memerag_judgement_{lang}.json"
        err_hint   = f"python judge_memerag.py --lang {lang} --all"
    else:
        input_path = _results_root() / lang / "qags_judgement.json"
        err_hint   = f"python llm-experiments/qags/judge_qags.py --dataset {lang} --all"

    if not input_path.exists():
        print(f"\n[ERROR] Judge output not found: {input_path}")
        print(f"  Run: {err_hint}")
        sys.exit(1)

    judge_data = load_json(input_path)
    records    = [r for r in judge_data.get("records", []) if isinstance(r, dict)]
    if dataset != "memerag":
        records = _normalize_qags_records(records)
    if not records:
        print(f"[ERROR] No records in {input_path}")
        sys.exit(1)

    if args.samples is not None:
        records = records[: args.samples]

    print(f"Records   : {len(records)} loaded from {input_path.name}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    check_proposers(records, proposers)
    random.seed(config.SEED)

    # Inter-judge pairwise kappa (before any aggregation)
    if len(proposers) >= 2:
        print(f"\n{'='*60}\nINTER-JUDGE AGREEMENT\n{'='*60}")
        print_interjudge_kappa(records, proposers)

    all_results:          list[tuple[str, float | None, float | None]] = []
    ds_reliability_result: dict | None = None
    algo_keys:            list[str] = []
    llm_preds:            dict[str, str | None] | None = None
    llm_row_label:        str = "llm_agg"
    retry_hint:           str | None = None

    # ── Step 1: Algorithmic aggregation ──────────────────────────────────
    if config.AGGREGATION_ALGOS:
        print(f"\n{'='*60}\nALGORITHMIC AGGREGATION\n{'='*60}")
        algo_rows, ds_reliability_result = run_algo_aggregation(records, lang, proposers)
        all_results.extend(algo_rows)
        algo_keys = list(config.AGGREGATION_ALGOS)

    # ── Step 2: LLM aggregation ───────────────────────────────────────────
    if config.AGGREGATOR_LLM:
        print(f"\n{'='*60}\nLLM AGGREGATION ({config.AGGREGATOR_LLM.upper()})\n{'='*60}")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        try:
            from google import genai
            from google.genai.types import HttpOptions
        except ImportError:
            genai = HttpOptions = None

        clients = init_clients(proposers)
        result  = run_llm_aggregation(
            records, lang, proposers, clients, genai, HttpOptions,
        )
        retry_hint: str | None = None
        if result:
            row_label, bacc, kappa, sid_to_label, retry_hint = result
            all_results.append((row_label, bacc, kappa))
            llm_preds     = sid_to_label
            llm_row_label = row_label

    # ── Summary table ─────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
        print_metrics_table(all_results)

    # ── Method pairwise kappa + CSV ───────────────────────────────────────
    print(f"\n{'='*60}\nOUTPUT FILES\n{'='*60}")
    if algo_keys or llm_preds:
        save_method_kappa_json(
            records, proposers, algo_keys, lang,
            llm_preds, llm_row_label, ds_reliability_result,
        )
    save_metrics_csv(
        records, proposers, algo_keys, lang,
        llm_label=llm_row_label if llm_preds else None,
        llm_sid_to_label=llm_preds,
        ds_reliability_result=ds_reliability_result,
    )

    if retry_hint:
        print(f"\n{'='*60}")
        print(f"  [ACTION NEEDED] Some records failed — re-run to retry:")
        print(f"    {retry_hint}")

if __name__ == "__main__":
    main()
