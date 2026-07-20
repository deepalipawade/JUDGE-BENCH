from __future__ import annotations

"""
Plot aggregation results for MEMERAG.

Produces two figures per language:
  1. Full pairwise Cohen's Kappa heatmap — all methods vs all methods
     (individual judges + majority vote + Gemma aggregator + OW-I + ISP + Dawid-Skene)
  2. Balanced accuracy bar chart — all methods vs gold label

Usage:
    python llm-experiments/memerag/plot_memerag_results.py --lang en
    python llm-experiments/memerag/plot_memerag_results.py --lang es --no-show
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]

LABELS_MEMERAG = {"Supported", "Not Supported"}
POS = "Supported"
NEG = "Not Supported"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    return [r for r in data.get("records", []) if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Prediction extractors
# ---------------------------------------------------------------------------

def get_individual_preds(records: list[dict], models: list[str]) -> dict[str, list[str | None]]:
    preds: dict[str, list[str | None]] = {m: [] for m in models}
    for rec in records:
        for m in models:
            label = rec.get("model_outputs", {}).get(m, {}).get("label")
            preds[m].append(label if label in LABELS_MEMERAG else None)
    return preds


def get_mv_preds(records: list[dict], tiebreak_models: list[str]) -> list[str | None]:
    result = []
    for rec in records:
        votes = [
            info.get("label")
            for info in rec.get("model_outputs", {}).values()
            if isinstance(info, dict)
        ]
        valid = [v for v in votes if v in LABELS_MEMERAG]
        if not valid:
            result.append(None)
            continue
        c = Counter(valid)
        top = c.most_common()
        if len(top) == 1 or top[0][1] != top[1][1]:
            result.append(top[0][0])
        else:
            picked = next(
                (rec.get("model_outputs", {}).get(m, {}).get("label")
                 for m in tiebreak_models
                 if rec.get("model_outputs", {}).get(m, {}).get("label") in LABELS_MEMERAG),
                None,
            )
            result.append(picked)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pairwise_kappa(a: list[str | None], b: list[str | None]) -> float | None:
    agree = total = 0
    ca: Counter = Counter()
    cb: Counter = Counter()
    for la, lb in zip(a, b):
        if la not in LABELS_MEMERAG or lb not in LABELS_MEMERAG:
            continue
        total += 1
        ca[la] += 1
        cb[lb] += 1
        if la == lb:
            agree += 1
    if total == 0:
        return None
    po = agree / total
    pe = sum((ca[l] / total) * (cb[l] / total) for l in LABELS_MEMERAG)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


def balanced_accuracy(preds: list[str | None], golds: list[str | None]) -> float | None:
    tp = tn = fp = fn = 0
    for p, g in zip(preds, golds):
        if p not in LABELS_MEMERAG or g not in LABELS_MEMERAG:
            continue
        if g == POS and p == POS:   tp += 1
        elif g == NEG and p == NEG: tn += 1
        elif g == NEG and p == POS: fp += 1
        else:                       fn += 1
    pos, neg = tp + fn, tn + fp
    return 0.5 * (tp / pos + tn / neg) if pos and neg else None


# ---------------------------------------------------------------------------
# Short display names
# ---------------------------------------------------------------------------

SHORT = {
    "gemini-2.5-flash":                    "Gemini-2.5-Flash",
    "google/gemma-4-26b-a4b-it-maas":      "Gemma-4-27B",
    "gemini-2.5-flash-lite":               "Gemini-2.5-Flash-Lite",
    "meta/llama-3.3-70b-instruct-maas":    "Llama-3.3-70B",
    "Majority Vote":                        "Majority Vote",
    "Gemma Aggregator":                     "Gemma Aggregator",
    "OW-I":                                 "OW-I",
    "ISP":                                  "ISP",
    "Dawid-Skene":                          "Dawid-Skene",
}

def short(name: str) -> str:
    return SHORT.get(name, name.split("/")[-1][:20])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MEMERAG aggregation results.")
    parser.add_argument("--lang", default="en", help="Language subfolder (en, es, de, ...)")
    parser.add_argument("--no-show", action="store_true", help="Save figures but do not display them")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    base = ROOT / "results_tmp" / "memerag_ext" / args.lang
    judgement_path = base / f"memerag_judgement_{args.lang}.json"
    gemma_path     = base / f"aggregator_gemma_{args.lang}.json"
    weighted_path  = base / f"aggregator_weighted_{args.lang}.json"

    if not judgement_path.exists():
        print(f"[ERROR] {judgement_path} not found")
        return

    judgement_records = load_records(judgement_path)
    n = len(judgement_records)
    print(f"Lang={args.lang}  |  {n} records")

    # Model order (display + tiebreak)
    MODEL_ORDER = [
        "gemini-2.5-flash",
        "google/gemma-4-26b-a4b-it-maas",
        "gemini-2.5-flash-lite",
        "meta/llama-3.3-70b-instruct-maas",
    ]
    available = set(judgement_records[0].get("model_outputs", {}).keys())
    models = [m for m in MODEL_ORDER if m in available] + \
             [m for m in available if m not in MODEL_ORDER]

    # Individual model predictions and tiebreak order
    ind_preds = get_individual_preds(judgement_records, models)
    golds     = [r.get("gold_label") for r in judgement_records]

    ind_baccs = {m: balanced_accuracy(ind_preds[m], golds) for m in models}
    tiebreak_models = sorted(models, key=lambda m: ind_baccs[m] or 0, reverse=True)

    # Majority vote
    mv_preds = get_mv_preds(judgement_records, tiebreak_models)

    # Collect all method predictions
    all_methods: dict[str, list[str | None]] = {**ind_preds, "Majority Vote": mv_preds}

    # Gemma aggregator
    if gemma_path.exists():
        agg_records = load_records(gemma_path)
        if len(agg_records) == n:
            all_methods["Gemma Aggregator"] = [
                r.get("aggregator_label") for r in agg_records
            ]
        else:
            print(f"[WARN] Gemma aggregator length mismatch ({len(agg_records)} vs {n}), skipping")
    else:
        print(f"[INFO] Gemma aggregator not found: {gemma_path}")

    # Weighted aggregators (OW-I, ISP, Dawid-Skene)
    if weighted_path.exists():
        w_records = load_records(weighted_path)
        if len(w_records) == n:
            # all_methods["OW-I"]        = [r.get("owi_label")           for r in w_records]
            # all_methods["ISP"]         = [r.get("isp_label")           for r in w_records]
            # all_methods["Dawid-Skene"] = [r.get("dawid_skene_label")   for r in w_records]
        # else:
            print(f"[WARN] Weighted JSON length mismatch ({len(w_records)} vs {n}), skipping")
    else:
        print(f"[INFO] Weighted aggregator not found: {weighted_path}")

    method_names = list(all_methods.keys())
    short_names  = [short(m) for m in method_names]

    # ── Figure 1: Pairwise Kappa Heatmap ─────────────────────────────────────
    k = len(method_names)
    kappa_mat = np.full((k, k), np.nan)
    for i, ma in enumerate(method_names):
        for j, mb in enumerate(method_names):
            if i == j:
                kappa_mat[i, j] = 1.0
            elif j > i:
                val = pairwise_kappa(all_methods[ma], all_methods[mb])
                kappa_mat[i, j] = val if val is not None else np.nan
                kappa_mat[j, i] = kappa_mat[i, j]

    fig1, ax1 = plt.subplots(figsize=(max(8, k * 1.1), max(6, k * 0.9)))
    mask = np.zeros_like(kappa_mat, dtype=bool)
    sns.heatmap(
        kappa_mat,
        ax=ax1,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Cohen's Kappa"},
    )
    ax1.set_title(
        f"Pairwise Cohen's Kappa — MEMERAG ({args.lang.upper()})\n"
        "(judges vs judges and aggregation methods)",
        fontsize=12, pad=12,
    )
    ax1.tick_params(axis="x", rotation=30, labelsize=9)
    ax1.tick_params(axis="y", rotation=0,  labelsize=9)
    fig1.tight_layout()

    heatmap_path = base / f"plot_kappa_heatmap_{args.lang}.png"
    fig1.savefig(heatmap_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {heatmap_path}")

    # ── Figure 2: Balanced Accuracy Bar Chart ─────────────────────────────────
    baccs = [balanced_accuracy(all_methods[m], golds) for m in method_names]

    # Colour by type
    type_colours = {
        "individual":  "#4C9BE8",
        "majority":    "#F5A623",
        "aggregator":  "#7ED321",
        "weighted":    "#9B59B6",
    }
    def method_type(name: str) -> str:
        if name in ind_preds:         return "individual"
        if name == "Majority Vote":   return "majority"
        if name == "Gemma Aggregator": return "aggregator"
        return "weighted"

    colours = [type_colours[method_type(m)] for m in method_names]

    fig2, ax2 = plt.subplots(figsize=(max(8, k * 0.9), 5))
    bars = ax2.bar(short_names, [b * 100 if b is not None else 0 for b in baccs], color=colours, edgecolor="white", width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, baccs):
        if val is not None:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val * 100:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

    ax2.set_ylabel("Balanced Accuracy (%)", fontsize=10)
    ax2.set_title(
        f"Balanced Accuracy vs Gold — MEMERAG ({args.lang.upper()})",
        fontsize=12, pad=10,
    )
    ax2.tick_params(axis="x", rotation=30, labelsize=9)
    ax2.set_ylim(0, 100)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_labels = {"Individual Judge": "individual", "Majority Vote": "majority",
                     "Gemma Aggregator": "aggregator", "Weighted (OW-I/ISP/DS)": "weighted"}
    handles = [Patch(color=type_colours[v], label=k) for k, v in legend_labels.items()
               if any(method_type(m) == v for m in method_names)]
    ax2.legend(handles=handles, fontsize=8, loc="lower right")

    fig2.tight_layout()
    barplot_path = base / f"plot_bacc_barplot_{args.lang}.png"
    fig2.savefig(barplot_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {barplot_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
