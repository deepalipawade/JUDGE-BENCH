"""
Cross-language comparison plots using the stacked BAcc/Kappa bar style.

Each bar = solid block (Kappa) + hatched extension (BAcc − Kappa gap).
Labels: BAcc% on top, κ= inside the solid block.

Generates 4 plots (2 method sets × 2 layouts):

  Layout A — x=languages, bars=methods (one grouped chart):
    crosslang_llm_vs_panel.png
    crosslang_algo_vs_llm.png

  Layout B — 5 subplots (3+2 grid), each subplot = one language, x=methods:
    crosslang_llm_vs_panel_subplots.png
    crosslang_algo_vs_llm_subplots.png

Usage:
    python llm-experiments/pipeline/plot_llm_algo_crosslang.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_tmp" / "memerag_ext"
OUT_DIR     = RESULTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGS       = ["en", "de", "es", "fr", "hi"]
LANG_LABELS = {"en": "EN", "de": "DE", "es": "ES", "fr": "FR", "hi": "HI"}

# ---------------------------------------------------------------------------
# Colors — match the per-language reference plot palette
# ---------------------------------------------------------------------------

COLORS = {
    "individual":   "#4878CF",   # blue
    "majority":     "#F4A936",   # orange
    "weighted_agg": "#3A8C5C",   # green
    "llm_agg":      "#C0392B",   # red
}

# Short display names for individual judges
JUDGE_SHORT = {
    "MiniMax-M2.7":                        "MiniMax",
    "GPT-OSS-120b":                        "GPT-OSS",
    "Qwen3.6-35B":                         "Qwen3.6",
    "gemini-2.5-flash":                    "Gemini",
    "meta/llama-3.3-70b-instruct-maas":    "LLaMA",
    "google/gemma-4-26b-a4b-it-maas":      "Gemma",
    "Apertus-8B":                          "Apertus",
}

# Canonical order for individual judges
JUDGE_ORDER = list(JUDGE_SHORT.keys())

# Per-language colours (used in Layout A only)
LANG_COLORS = {
    "en": "#4878CF", "de": "#E08030", "es": "#3A8C5C",
    "fr": "#9467BD", "hi": "#C44E52",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse(val: str) -> float | None:
    if not val or val.strip() in ("", "None", "-"):
        return None
    try:
        return float(val.strip().rstrip("%"))
    except ValueError:
        return None


BEST_JUDGE_COLOR = "#8B0000"   # dark red — distinct from llm_agg red


def load_metrics(lang: str) -> dict[str, dict]:
    """Return {key: {bacc, kappa, color}} for one language."""
    path = RESULTS_DIR / lang / f"metrics_summary_{lang}.csv"
    if not path.exists():
        print(f"  [warn] missing: {path}", file=sys.stderr)
        return {}

    out: dict[str, dict] = {}

    def _add(key: str, bacc, kappa, color: str) -> None:
        if bacc is None and kappa is None:
            return
        if key not in out or (bacc or 0) > (out[key]["bacc"] or 0):
            out[key] = {"bacc": bacc, "kappa": kappa, "color": color}

    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            kind  = (row.get("kind")  or "").strip()
            name  = (row.get("name")  or "").strip()
            bacc  = _parse(row.get("balanced_accuracy", ""))
            kappa = _parse(row.get("cohen_kappa", ""))

            if kind == "individual" and name in JUDGE_SHORT:
                _add(f"judge_{name}", bacc, kappa, COLORS["individual"])
            elif kind == "majority_vote" and name == "majority":
                _add("majority", bacc, kappa, COLORS["majority"])
            elif kind == "weighted_agg":
                lname = name.lower()
                key = {"ow-i": "owi", "dawid-skene": "ds"}.get(lname, lname)
                if key in ("owi", "isp", "iwmv", "mace", "ds"):
                    _add(key, bacc, kappa, COLORS["weighted_agg"])
            elif kind == "aggregator":
                lname = name.lower()
                if "random" in lname:
                    _add("llm_random", bacc, kappa, COLORS["llm_agg"])
                elif "best_bacc" in lname:
                    _add("llm_best_bacc", bacc, kappa, COLORS["llm_agg"])

    return out


def load_all() -> dict[str, dict[str, dict]]:
    all_data = {lang: load_metrics(lang) for lang in LANGS}

    # Find the individual judge with the highest AVERAGE BAcc across all languages,
    # then add it as "best_judge" entry per language for use in plots.
    judge_avg: dict[str, list[float]] = {}
    for lang in LANGS:
        for m in JUDGE_ORDER:
            key = f"judge_{m}"
            entry = all_data[lang].get(key)
            if entry and entry["bacc"] is not None:
                judge_avg.setdefault(m, []).append(entry["bacc"])

    best_model = max(judge_avg, key=lambda m: sum(judge_avg[m]) / len(judge_avg[m]))
    best_avg   = sum(judge_avg[best_model]) / len(judge_avg[best_model])
    print(f"  Best individual judge: {best_model}  (avg BAcc = {best_avg:.2f}%)")

    for lang in LANGS:
        src = all_data[lang].get(f"judge_{best_model}")
        if src:
            all_data[lang]["best_judge"] = {
                "bacc":  src["bacc"],
                "kappa": src["kappa"],
                "color": BEST_JUDGE_COLOR,
                "model": best_model,
            }

    return all_data, best_model


# ---------------------------------------------------------------------------
# Single stacked bar
# ---------------------------------------------------------------------------

def draw_bar(ax: plt.Axes, x: float, width: float,
             bacc: float | None, kappa: float | None, color: str,
             fontsize_top: float = 7.0, fontsize_inner: float = 6.5) -> None:
    if bacc is None:
        return
    kappa = kappa or 0.0

    ax.bar(x, kappa, width=width, color=color, alpha=0.90,
           edgecolor="white", linewidth=0.5, zorder=3)

    gap = bacc - kappa
    if gap > 0:
        ax.bar(x, gap, bottom=kappa, width=width,
               color=color, alpha=0.28, hatch="////",
               edgecolor=color, linewidth=0.0, zorder=3)

    ax.text(x, bacc + 0.8, f"{bacc:.1f}%",
            ha="center", va="bottom", fontsize=fontsize_top,
            fontweight="bold", color="#222222", zorder=5)

    if kappa > 14:
        ax.text(x, kappa / 2, f"κ={kappa:.1f}",
                ha="center", va="center", fontsize=fontsize_inner,
                color="white", fontweight="bold", zorder=5)


# ---------------------------------------------------------------------------
# Layout A: one grouped chart, x=languages, bars=methods
# ---------------------------------------------------------------------------

def make_plot_by_lang(
    series: list[tuple[str, str]],
    all_data: dict[str, dict[str, dict]],
    title: str,
    out_path: Path,
) -> None:
    n_langs  = len(LANGS)
    n_series = len(series)
    group_gap = 0.9
    bar_width = 0.65 / n_series

    group_centres = np.arange(n_langs) * (1.0 + group_gap)
    offsets = [(i - (n_series - 1) / 2) * (bar_width + 0.04) for i in range(n_series)]

    fig, ax = plt.subplots(figsize=(max(10, n_langs * n_series * 0.55 + 3), 5.5))

    for gi, lang in enumerate(LANGS):
        for si, (key, _) in enumerate(series):
            entry = all_data[lang].get(key)
            if entry:
                draw_bar(ax, group_centres[gi] + offsets[si], bar_width,
                         entry["bacc"], entry["kappa"], entry["color"])

    ax.set_xticks(group_centres)
    ax.set_xticklabels([LANG_LABELS[l] for l in LANGS], fontsize=12, fontweight="bold")
    ax.set_xlim(group_centres[0] - 0.8, group_centres[-1] + 0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("(%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axhline(50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for gi in range(n_langs - 1):
        mid = (group_centres[gi] + group_centres[gi + 1]) / 2
        ax.axvline(mid, color="#cccccc", linewidth=0.7, linestyle=":", zorder=1)

    handles = []
    seen: set[str] = set()
    for key, label in series:
        color = next((all_data[l][key]["color"] for l in LANGS if key in all_data[l]), "#aaa")
        if color not in seen:
            seen.add(color)
        handles.append(mpatches.Patch(color=color, alpha=0.9, label=label))
    handles.append(mpatches.Patch(facecolor="#aaaaaa", alpha=0.28, hatch="////",
                                  edgecolor="#aaaaaa", label="Gap (BAcc − Kappa)"))
    ax.legend(handles=handles, fontsize=9, framealpha=0.85, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Layout B: 5 subplots in 3+2 grid, each = one language, x=methods
# ---------------------------------------------------------------------------

def _draw_subplot(ax: plt.Axes, lang: str, lang_data: dict[str, dict],
                  series: list[tuple[str, str]], bar_width: float) -> None:
    n = len(series)
    xs = np.arange(n)

    for i, (key, _) in enumerate(series):
        entry = lang_data.get(key)
        if entry:
            draw_bar(ax, xs[i], bar_width,
                     entry["bacc"], entry["kappa"], entry["color"],
                     fontsize_top=6.5, fontsize_inner=5.8)

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in series], fontsize=8,
                       rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("(%)", fontsize=8)
    ax.set_title(LANG_LABELS[lang], fontsize=11, fontweight="bold", pad=5)
    ax.axhline(50, color="#888888", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_subplots(
    series: list[tuple[str, str]],
    all_data: dict[str, dict[str, dict]],
    title: str,
    out_path: Path,
) -> None:
    """3+2 grid of subplots — one per language, matching the heatmap reference layout."""
    # Layout: row 0 = EN, ES, DE  |  row 1 = FR, HI (centred)
    layout = [["en", "es", "de"], ["fr", "hi"]]

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    bar_width = 0.55

    # Top row: 3 subplots
    top_axes = fig.add_subplot(2, 3, 1), fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3)
    for ax, lang in zip(top_axes, layout[0]):
        _draw_subplot(ax, lang, all_data[lang], series, bar_width)

    # Bottom row: 2 subplots centred (use columns 1 and 2 of a 3-column grid)
    bot_axes = fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5)
    for ax, lang in zip(bot_axes, layout[1]):
        _draw_subplot(ax, lang, all_data[lang], series, bar_width)

    # Shared legend below the figure
    handles = []
    seen: set[str] = set()
    for key, label in series:
        color = next((all_data[l][key]["color"] for l in LANGS if key in all_data[l]), "#aaa")
        if color not in seen:
            seen.add(color)
        handles.append(mpatches.Patch(color=color, alpha=0.9, label=label))
    handles.append(mpatches.Patch(facecolor="#aaaaaa", alpha=0.28, hatch="////",
                                  edgecolor="#aaaaaa", label="Gap (BAcc − Kappa)"))
    fig.legend(handles=handles, fontsize=9, framealpha=0.85,
               loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Layout C: single plot with averaged values across all languages
# ---------------------------------------------------------------------------

def make_averaged_plot(
    series: list[tuple[str, str]],
    all_data: dict[str, dict[str, dict]],
    title: str,
    out_path: Path,
    ref_lines: list[tuple[float, str, str]] | None = None,
) -> None:
    """ref_lines: list of (value, label, color) — drawn as horizontal dashed lines."""
    """Single stacked bar chart where each bar = mean BAcc / mean Kappa across all languages."""
    keys   = [key for key, _ in series]
    labels = [lbl for _, lbl in series]
    n      = len(series)
    bar_width = 0.55
    xs = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1 + 2), 5.5))

    for i, (key, _) in enumerate(series):
        baccs  = [all_data[l][key]["bacc"]  for l in LANGS if key in all_data[l] and all_data[l][key]["bacc"]  is not None]
        kappas = [all_data[l][key]["kappa"] for l in LANGS if key in all_data[l] and all_data[l][key]["kappa"] is not None]
        if not baccs:
            continue
        avg_bacc  = sum(baccs)  / len(baccs)
        avg_kappa = sum(kappas) / len(kappas) if kappas else 0.0
        color = next((all_data[l][key]["color"] for l in LANGS if key in all_data[l]), "#aaaaaa")
        draw_bar(ax, xs[i], bar_width, avg_bacc, avg_kappa, color,
                 fontsize_top=8.0, fontsize_inner=7.0)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("(%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axhline(50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

    # Draw reference lines (e.g. best individual judge ceiling)
    ref_handles = []
    if ref_lines:
        for ref_val, ref_label, ref_color in ref_lines:
            ax.axhline(ref_val, color=ref_color, linestyle="--", linewidth=1.5,
                       zorder=6, alpha=0.9)
            from matplotlib.lines import Line2D
            ref_handles.append(Line2D([0], [0], color=ref_color, linestyle="--",
                                      linewidth=1.5, label=ref_label))

    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.01, 0.98, "Values averaged across EN / DE / ES / FR / HI",
            transform=ax.transAxes, fontsize=8, color="#666666",
            va="top", ha="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------

# Plot 1: all 7 individual judges + the two LLM aggregator strategies
PLOT1_SERIES = (
    [(f"judge_{m}", JUDGE_SHORT[m]) for m in JUDGE_ORDER]
    + [
        ("llm_best_bacc", "Best-BAcc\nLLM"),
        ("llm_random",    "Random\nLLM"),
    ]
)

PLOT2_SERIES = [
    ("majority",      "Majority"),
    ("owi",           "OWI"),
    ("isp",           "ISP"),
    ("iwmv",          "IWMV"),
    ("ds",            "Dawid-\nSkene"),
    ("mace",          "MACE"),
    ("llm_best_bacc", "Best-BAcc\nLLM"),
    ("best_judge",    "Best\nJudge"),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading metrics...")
    all_data, best_judge_model = load_all()
    for lang in LANGS:
        found = [k for k, v in all_data[lang].items() if v.get("bacc") is not None]
        print(f"  [{lang.upper()}] {found}")

    # Compute best judge avg BAcc for the reference line
    best_judge_baccs = [
        all_data[l]["best_judge"]["bacc"]
        for l in LANGS if "best_judge" in all_data[l] and all_data[l]["best_judge"]["bacc"] is not None
    ]
    best_judge_avg = sum(best_judge_baccs) / len(best_judge_baccs) if best_judge_baccs else None
    short_name = JUDGE_SHORT.get(best_judge_model, best_judge_model.split("/")[-1])
    ref_line = [(best_judge_avg, f"Best judge ({short_name})", BEST_JUDGE_COLOR)] if best_judge_avg else None
    # short_name already trimmed; label used in legend only (no inline text)

    make_subplots(
        PLOT1_SERIES, all_data,
        title="Balanced Accuracy & Kappa — Panel Majority vs LLM Aggregators",
        out_path=OUT_DIR / "crosslang_llm_vs_panel_subplots.png",
    )
    make_subplots(
        PLOT2_SERIES, all_data,
        title="Balanced Accuracy & Kappa — Aggregation Strategies Comparison",
        out_path=OUT_DIR / "crosslang_algo_vs_llm_subplots.png",
    )

    make_averaged_plot(
        PLOT1_SERIES, all_data,
        title="Balanced Accuracy & Kappa — Panel Majority vs LLM Aggregators (avg. across languages)",
        out_path=OUT_DIR / "crosslang_llm_vs_panel_avg.png",
    )
    make_averaged_plot(
        PLOT2_SERIES, all_data,
        title="Balanced Accuracy & Kappa — Aggregation Strategies Comparison (avg. across languages)",
        out_path=OUT_DIR / "crosslang_algo_vs_llm_avg.png",
        ref_lines=ref_line,
    )


if __name__ == "__main__":
    main()
