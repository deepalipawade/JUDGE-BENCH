"""
Visualize pipeline results: inter-judge kappa (6 plot types) + balanced accuracy.

Reads defaults from config.py. Override per-run with CLI flags.

Usage:
    python pipeline/visualize.py
    python pipeline/visualize.py --lang de
    python pipeline/visualize.py --lang en de fr
    python pipeline/visualize.py --all
    python pipeline/visualize.py --lang en --aggregator_llm ds_rank

Optional dependencies:
    pip install seaborn pandas networkx   (for clustermap + network plots)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
PIPELINE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PIPELINE_DIR))
from utils.io import load_json


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    spec = importlib.util.spec_from_file_location("config", PIPELINE_DIR / "config.py")
    cfg  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(cfg)                    # type: ignore[union-attr]
    return {
        "lang":          getattr(cfg, "LANG", "en"),
        "proposers":     list(getattr(cfg, "PROPOSERS", [])),
        "ignore_models": set(getattr(cfg, "IGNORE_MODELS", set())),
    }


def _short(model: str) -> str:
    return model.split("/")[-1][:22]


# ---------------------------------------------------------------------------
# Shared helper — build NxN kappa matrix
# ---------------------------------------------------------------------------

def _kappa_matrix(models: list[str], method_kappa: dict) -> list[list[float]]:
    n = len(models)
    mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i == j:
                continue
            k = method_kappa.get(f"{ma} vs {mb}") or method_kappa.get(f"{mb} vs {ma}")
            if k is not None:
                mat[i][j] = k
    return mat


def _classical_mds(dist_matrix: list[list[float]]) -> list[tuple[float, float]]:
    """Pure-numpy classical MDS — no sklearn needed."""
    import numpy as np
    D = np.array(dist_matrix, dtype=float)
    n = D.shape[0]
    D2 = D ** 2
    J  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx  = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    coords = vecs[:, :2] * np.sqrt(np.maximum(vals[:2], 0))
    return [(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]




# ---------------------------------------------------------------------------
# Shared helpers for combined kappa plots (judges + algo methods + LLM aggs)
# ---------------------------------------------------------------------------

_ALGO_KEYS  = {"majority", "owi", "isp", "ds"}
_ALGO_ORDER = ["majority", "owi", "isp", "ds"]

# Map both lowercase and display-name variants → canonical key
_ALGO_NORM: dict[str, str] = {
    "majority": "majority", "MV":          "majority",
    "owi":      "owi",      "OW-I":        "owi",
    "isp":      "isp",      "ISP":         "isp",
    "ds":       "ds",       "Dawid-Skene": "ds",
}
# All name variants per canonical (for lookup across inconsistent JSON keys)
_ALGO_VARIANTS: dict[str, list[str]] = {
    "majority": ["majority", "MV"],
    "owi":      ["owi", "OW-I"],
    "isp":      ["isp", "ISP"],
    "ds":       ["ds", "Dawid-Skene"],
}
_ALGO_LABEL = {"majority": "MajVote", "owi": "OWI", "isp": "ISP", "ds": "Dawid-Skene"}


def _norm_entity(k: str) -> str:
    return _ALGO_NORM.get(k, k)


def _cat_entity(k: str) -> str:
    if k in _ALGO_KEYS:       return "algo"
    if k.startswith("llm_"):  return "llm"
    return "judge"


def _display_entity(k: str) -> str:
    if k in _ALGO_LABEL:                return _ALGO_LABEL[k]
    if k == "llm_random":               return "LLM-Random"
    if k.startswith("llm_best_bacc("):  return "LLM-BestBAcc"
    if k.startswith("llm_worse_bacc("): return "LLM-WorseBAcc"
    if k.startswith("llm_ds_rank("):    return "LLM-DSRank"
    return _short(k)


def _kappa_lookup(method_kappa: dict, a: str, b: str) -> float | None:
    """Lookup kappa handling both canonical and display-name algo variants."""
    a_names = _ALGO_VARIANTS.get(a, [a])
    b_names = _ALGO_VARIANTS.get(b, [b])
    for an in a_names:
        for bn in b_names:
            v = method_kappa.get(f"{an} vs {bn}") or method_kappa.get(f"{bn} vs {an}")
            if v is not None:
                return float(v)
    return None


def _discover_methods(method_kappa: dict) -> tuple[list[str], list[str]]:
    """Return (known_algos, known_llms) discovered from JSON keys."""
    seen_algo: set[str] = set()
    seen_llm:  set[str] = set()
    for key in method_kappa:
        for part in key.split(" vs "):
            p = _norm_entity(part.strip())
            if p in _ALGO_KEYS:        seen_algo.add(p)
            elif p.startswith("llm_"): seen_llm.add(p)
    algos = [a for a in _ALGO_ORDER if a in seen_algo]
    llms  = sorted(seen_llm, key=lambda k: (
        0 if "random" in k else (1 if "best_bacc" in k else (2 if "worse_bacc" in k else 3))
    ))
    return algos, llms


# ---------------------------------------------------------------------------
# Combined kappa — ranked horizontal bar (judges + algo + LLM methods)
# ---------------------------------------------------------------------------

def plot_kappa_combined_ranked(
    lang: str, models: list[str], method_kappa: dict, plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError as e:
        print(f"  [SKIP] combined ranked — missing {e}")
        return

    model_set           = set(models)
    known_algo, known_llm = _discover_methods(method_kappa)
    known_algo_set      = set(known_algo)
    known_llm_set       = set(known_llm)

    pairs: list[tuple[str, float]] = []
    seen:  set[tuple[str, str]]    = set()

    for raw_key, val in method_kappa.items():
        parts = raw_key.split(" vs ", 1)
        if len(parts) != 2:
            continue
        a = _norm_entity(parts[0].strip())
        b = _norm_entity(parts[1].strip())
        cat_a, cat_b = _cat_entity(a), _cat_entity(b)

        # Keep only entities we know about
        if cat_a == "judge" and a not in model_set:       continue
        if cat_b == "judge" and b not in model_set:       continue
        if cat_a == "algo"  and a not in known_algo_set:  continue
        if cat_b == "algo"  and b not in known_algo_set:  continue
        if cat_a == "llm"   and a not in known_llm_set:   continue
        if cat_b == "llm"   and b not in known_llm_set:   continue

        pair_key = tuple(sorted([a, b]))
        if pair_key in seen:
            continue
        seen.add(pair_key)

        lbl = f"{_display_entity(a)}  vs  {_display_entity(b)}"
        pairs.append((lbl, float(val)))

    if not pairs:
        print(f"  [SKIP] combined ranked — no pairs found")
        return

    pairs.sort(key=lambda x: x[1])
    labels, kappas = zip(*pairs)

    cmap   = matplotlib.colormaps["RdYlGn"]
    colors = [cmap(k) for k in kappas]

    fig, ax = plt.subplots(figsize=(9, max(4, len(pairs) * 0.32)))
    bars = ax.barh(range(len(pairs)), kappas, color=colors, edgecolor="white",
                   linewidth=0.4)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Cohen's Kappa")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.6, color="grey", linestyle="--", linewidth=0.7, alpha=0.5,
               label="κ = 0.6")
    ax.legend(fontsize=8)

    for bar, k in zip(bars, kappas):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{k:.2f}", va="center", fontsize=7)

    ax.set_title(f"Pairwise Kappa — All pairs ranked  [{lang.upper()}]",
                 fontsize=12, pad=10)
    fig.tight_layout()
    out = plots_dir / f"kappa_combined_ranked_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Combined kappa — faceted (one subplot per judge + per method)
# ---------------------------------------------------------------------------

def plot_kappa_combined_faceted(
    lang: str, models: list[str], method_kappa: dict, plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] combined faceted — missing {e}")
        return

    known_algo, known_llm = _discover_methods(method_kappa)
    all_entities = models + known_algo + known_llm
    n            = len(all_entities)

    palette      = plt.cm.tab10.colors
    entity_color = {e: palette[i % len(palette)] for i, e in enumerate(all_entities)}

    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3.5, rows * 3),
                              sharey=False)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, entity in enumerate(all_entities):
        ax      = axes_list[idx]
        others  = [e for e in all_entities if e != entity]

        pairs_here: list[tuple[float, str, str]] = []
        for other in others:
            v = _kappa_lookup(method_kappa, entity, other)
            if v is not None:
                pairs_here.append((v, _display_entity(other),
                                   entity_color.get(other, "#888")))

        pairs_here.sort(reverse=True)
        ax.set_title(_display_entity(entity), fontsize=9, fontweight="bold")

        if not pairs_here:
            continue

        kaps, names, colors = zip(*pairs_here)
        ax.barh(range(len(names)), kaps, color=colors, edgecolor="white",
                linewidth=0.4)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.6, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("κ", fontsize=7)

    for idx in range(n, rows * cols):
        axes_list[idx].set_visible(False)

    fig.suptitle(f"Kappa — Per-entity view (judges + methods)  [{lang.upper()}]",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = plots_dir / f"kappa_combined_faceted_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Kappa plot 9 — MDS scatter (models projected to 2D by agreement distance)
# ---------------------------------------------------------------------------

def plot_kappa_mds(
    lang: str, models: list[str], method_kappa: dict, plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError as e:
        print(f"  [SKIP] MDS — missing {e}")
        return

    mat   = _kappa_matrix(models, method_kappa)
    names = [_short(m) for m in models]
    n     = len(models)

    dist  = [[1.0 - mat[i][j] for j in range(n)] for i in range(n)]
    coords = _classical_mds(dist)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Color = mean kappa with others
    mean_kappas = [
        sum(mat[i][j] for j in range(n) if j != i) / max(n - 1, 1)
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xs, ys, c=mean_kappas, cmap="RdYlGn",
                    vmin=0, vmax=1, s=150, edgecolors="#333", linewidth=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label="Mean kappa with others", shrink=0.8)

    for x, y, name in zip(xs, ys, names):
        txt = ax.text(x, y + (max(ys) - min(ys)) * 0.04, name,
                      ha="center", fontsize=8)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    ax.set_xlabel("MDS dimension 1 (agreement distance)")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title(
        f"Inter-judge Kappa — MDS projection  [{lang.upper()}]\n"
        f"(closer = more agreement)",
        fontsize=11, pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = plots_dir / f"interjudge_kappa_mds_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")




# ---------------------------------------------------------------------------
# Balanced accuracy — Option A: grouped bars (algo cluster)
# ---------------------------------------------------------------------------

def plot_bacc_grouped(
    lang: str, csv_rows: list[dict[str, str]], plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"  [SKIP] bacc grouped — missing {e}")
        return

    KIND_COLOR = {
        "individual":    "#4C9BE8",
        "majority_vote": "#F5A623",
        "weighted_agg":  "#5BAD6F",
        "aggregator":    "#D05050",
    }
    GROUPS = ["individual", "majority_vote", "weighted_agg", "aggregator"]

    # Parse rows
    by_kind: dict[str, list[tuple[str, float]]] = {k: [] for k in GROUPS}
    for row in csv_rows:
        kind = row.get("kind", "")
        if kind not in GROUPS:
            continue
        raw = row.get("balanced_accuracy") or ""
        try:
            val = float(raw.rstrip("%"))
        except ValueError:
            continue
        by_kind[kind].append((row.get("name", "?"), val))

    # Build x positions — advance x BETWEEN items only (not after the last),
    # then add GROUP_GAP before each new group. This keeps inter-group gap = GROUP_GAP exactly.
    GROUP_GAP     = 0.7
    CLUSTER_KINDS = {"weighted_agg", "aggregator"}
    ALGO_WIDTH    = 0.35  # spacing within cluster groups
    bar_x, bar_h, bar_c, bar_lbl = [], [], [], []
    group_x_ranges: dict[str, tuple[float, float]] = {}  # kind → (x_first, x_last)

    x = 0.0
    for kind in GROUPS:
        items = by_kind[kind]
        if not items:
            continue
        if bar_x:
            x += GROUP_GAP
        x_first = x
        spacing = ALGO_WIDTH if kind in CLUSTER_KINDS else 0.45
        for i, (name, val) in enumerate(items):
            if i > 0:
                x += spacing
            bar_x.append(x)
            bar_h.append(val)
            bar_lbl.append(name)
            bar_c.append(KIND_COLOR[kind])
        group_x_ranges[kind] = (x_first, x)

    if not bar_x:
        print(f"  [SKIP] bacc grouped — no data")
        return

    fig, ax = plt.subplots(figsize=(max(9, len(bar_x) * 0.8 + 2), 5))
    bars = ax.bar(bar_x, bar_h, color=bar_c, width=0.3,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(50.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Group background shading — use actual bar positions for bounds
    KIND_BG = {
        "individual":    "#e8f0fa",
        "majority_vote": "#fef4e0",
        "weighted_agg":  "#e8f6ed",
        "aggregator":    "#fce8e8",
    }
    for kind, (x_lo, x_hi) in group_x_ranges.items():
        ax.axvspan(x_lo - 0.45, x_hi + 0.45, alpha=0.25,
                   color=KIND_BG.get(kind, "#eee"), zorder=0)

    # Value labels
    for bar, val in zip(bars, bar_h):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_lbl, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=10)
    ax.set_ylim(40, min(102, max(bar_h) + 10))
    ax.set_title(f"Balanced Accuracy — Grouped by method  [{lang.upper()}]",
                 fontsize=12, pad=12)

    legend_patches = [
        mpatches.Patch(color="#4C9BE8", label="Individual judge"),
        mpatches.Patch(color="#F5A623", label="Majority vote"),
        mpatches.Patch(color="#5BAD6F", label="Weighted agg (OWI / ISP / DS)"),
        mpatches.Patch(color="#D05050", label="LLM aggregator"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = plots_dir / f"balanced_accuracy_grouped_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# BAcc + Kappa stacked bar (separate plot)
# bottom segment = kappa, hatched top = gap (bacc − kappa), total = bacc
# ---------------------------------------------------------------------------

def plot_bacc_kappa_stacked(
    lang: str, csv_rows: list[dict[str, str]], plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"  [SKIP] bacc+kappa stacked — missing {e}")
        return

    KIND_COLOR = {
        "individual":    "#4C9BE8",
        "majority_vote": "#F5A623",
        "weighted_agg":  "#5BAD6F",
        "aggregator":    "#D05050",
    }
    GROUPS      = ["individual", "majority_vote", "weighted_agg", "aggregator"]

    by_kind: dict[str, list[tuple[str, float, float]]] = {k: [] for k in GROUPS}
    for row in csv_rows:
        kind = row.get("kind", "")
        if kind not in GROUPS:
            continue
        try:
            bacc  = float((row.get("balanced_accuracy") or "").rstrip("%"))
            kappa = float((row.get("cohen_kappa")       or "").rstrip("%"))
        except ValueError:
            continue
        by_kind[kind].append((row.get("name", "?"), bacc, kappa))

    BAR_W   = 0.08   # thin bars
    SPACING = 0.13   # uniform center-to-center for ALL bars (within and across groups)
    bar_x, bar_kappa, bar_gap, bar_color, bar_lbl = [], [], [], [], []
    group_x_ranges: dict[str, tuple[float, float]] = {}

    x = 0.0
    for kind in GROUPS:
        items = by_kind[kind]
        if not items:
            continue
        if bar_x:
            x += SPACING   # same gap crossing a group boundary as within one
        x_first = x
        for i, (name, bacc, kappa) in enumerate(items):
            if i > 0:
                x += SPACING
            bar_x.append(x)
            bar_kappa.append(kappa)
            bar_gap.append(max(0.0, bacc - kappa))
            bar_lbl.append(name)
            bar_color.append(KIND_COLOR[kind])
        group_x_ranges[kind] = (x_first, x)

    if not bar_x:
        print(f"  [SKIP] bacc+kappa stacked — no data")
        return

    fig, ax = plt.subplots(figsize=(max(9, len(bar_x) * 0.8 + 2), 5))

    # Solid bottom = kappa
    ax.bar(bar_x, bar_kappa, color=bar_color, width=BAR_W,
           edgecolor="white", linewidth=0.5)

    # Hatched top = gap (bacc − kappa)
    ax.bar(bar_x, bar_gap, bottom=bar_kappa, color=bar_color, width=BAR_W,
           edgecolor="white", linewidth=0.5, alpha=0.35, hatch="///")

    ax.axhline(50.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    KIND_BG = {
        "individual":    "#e8f0fa",
        "majority_vote": "#fef4e0",
        "weighted_agg":  "#e8f6ed",
        "aggregator":    "#fce8e8",
    }
    pad = BAR_W / 2 + 0.05
    for kind, (x_lo, x_hi) in group_x_ranges.items():
        ax.axvspan(x_lo - pad, x_hi + pad, alpha=0.25,
                   color=KIND_BG.get(kind, "#eee"), zorder=0)

    # Midpoint tick on each bar — where 50% of that bar's height falls
    # solid (kappa) reaching above the tick = kappa > gap = genuinely earned
    # half_w = 0.18
    # for bx, kap, gap in zip(bar_x, bar_kappa, bar_gap):
    #     mid = (kap + gap) / 2
    #     ax.hlines(mid, bx - half_w, bx + half_w,
    #               colors="#888888", linewidths=1.0, alpha=0.6, zorder=4)

    # Annotate total (bacc) at top, kappa value at the solid/gap boundary
    for bx, kap, gap in zip(bar_x, bar_kappa, bar_gap):
        total = kap + gap
        ax.text(bx, total + 0.4, f"{total:.2f}%",
                ha="center", va="bottom", fontsize=7)
        ax.text(bx, kap, f"κ={kap:.2f}",
                ha="center", va="bottom", fontsize=5.5, color="#444444")

    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_lbl, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("(%)", fontsize=10)
    ax.set_ylim(0, min(105, max(k + g for k, g in zip(bar_kappa, bar_gap)) + 12))
    ax.set_title(
        f"Balanced Accuracy & Kappa — Grouped by method  [{lang.upper()}]",
        fontsize=11, pad=12,
    )

    legend_patches = [
        mpatches.Patch(color="#4C9BE8", label="Individual judge"),
        mpatches.Patch(color="#F5A623", label="Majority vote"),
        mpatches.Patch(color="#5BAD6F", label="Weighted agg"),
        mpatches.Patch(color="#D05050", label="LLM aggregator"),
        mpatches.Patch(facecolor="white", edgecolor="grey",
                       hatch="///", label="Gap (BAcc − Kappa)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=True, framealpha=0.9)

    fig.tight_layout()
    out = plots_dir / f"bacc_kappa_stacked_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Consensus analysis plots (vote splits + dissenter identity)
# ---------------------------------------------------------------------------

def plot_consensus_analysis(lang: str, data: dict, plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] consensus plots — missing {e}")
        return

    summary = data.get("summary", {})
    if not summary:
        print(f"  [SKIP] consensus plots — empty summary")
        return

    vote_splits   = summary.get("vote_split_distribution", {})
    unani_wrong   = summary.get("unanimous_wrong", {})
    dissenters    = summary.get("dissenter_counts", {})
    contrarians   = summary.get("correct_contrarians", {})
    by_model_dis  = dissenters.get("by_model", {})
    by_model_con  = contrarians.get("by_model", {})
    by_split_dis  = dissenters.get("by_split", {})
    n_samples     = summary.get("n_samples", 1)
    n_wrong       = unani_wrong.get("n_cases", 0)
    n_unan        = summary.get("n_unanimous", 0)

    # ── Figure 1: Vote split distribution ────────────────────────────────────
    if vote_splits:
        splits = list(vote_splits.keys())
        counts = [vote_splits[s] for s in splits]
        # Color: greenish for high agreement, orange for close
        split_colors = []
        for s in splits:
            hi = int(s.split("-")[0])
            total = hi + int(s.split("-")[1])
            ratio = hi / total if total else 1
            if ratio == 1.0:   split_colors.append("#4CAF50")   # unanimous — green
            elif ratio > 0.75: split_colors.append("#8BC34A")   # strong majority
            elif ratio > 0.6:  split_colors.append("#FFC107")   # moderate
            else:              split_colors.append("#FF7043")    # close split

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(splits, counts, color=split_colors, edgecolor="white", linewidth=0.6, width=0.55)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom", fontsize=9)

        # Annotate unanimous-wrong on the unanimous bar
        if n_wrong and splits and splits[0].endswith("-0"):
            ax.text(0, counts[0] / 2,
                    f"{n_wrong} wrong\n({n_wrong/n_unan*100:.1f}% of\nunanim.)",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")

        ax.set_xlabel("Vote split (majority–minority)", fontsize=10)
        ax.set_ylabel("Number of samples", fontsize=10)
        ax.set_title(f"Panel vote split distribution  [{lang.upper()}]\n"
                     f"n={n_samples}  |  {summary.get('pct_unanimous',0):.1f}% unanimous"
                     f"  |  {summary.get('pct_majority_correct',0):.1f}% majority correct",
                     fontsize=10, pad=10)
        ax.set_ylim(0, max(counts) * 1.18)
        fig.tight_layout()
        out = plots_dir / f"consensus_vote_splits_{lang}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")

    # ── Figure 2: Dissenter profile per model ────────────────────────────────
    if by_model_dis:
        models   = list(by_model_dis.keys())
        # Sort by minority votes descending
        models   = sorted(models, key=lambda m: by_model_dis[m].get("n_minority_votes", 0), reverse=True)
        short    = [_short(m) for m in models]
        n_dis    = [by_model_dis[m].get("n_minority_votes", 0)    for m in models]
        n_con    = [by_model_con.get(m, {}).get("n_correct_vs_majority", 0) for m in models]

        palette  = plt.cm.tab10.colors
        model_color = {m: palette[i % len(palette)] for i, m in enumerate(models)}

        fig, axes = plt.subplots(1, 2, figsize=(11, max(3, len(models) * 0.55 + 1.5)))

        # Left: how often each model is in the minority
        ax = axes[0]
        bars = ax.barh(short, n_dis,
                       color=[model_color[m] for m in models],
                       edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, n_dis):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=8)
        ax.set_xlabel("Times in minority vote", fontsize=9)
        ax.set_title("Who disagrees most?", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, max(n_dis) * 1.2 if n_dis else 1)

        # Right: of those dissents, how often were they RIGHT
        ax2 = axes[1]
        bars2 = ax2.barh(short, n_con,
                         color=[model_color[m] for m in models],
                         edgecolor="white", linewidth=0.4)
        for bar, v, nd in zip(bars2, n_con, n_dis):
            rate = f"  {v/nd*100:.0f}%" if nd else ""
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{v}{rate}", va="center", fontsize=8)
        ax2.set_xlabel("Times correct when majority wrong", fontsize=9)
        ax2.set_title("Who is right when dissenting?", fontsize=10, fontweight="bold")
        ax2.invert_yaxis()
        ax2.set_xlim(0, max(n_con) * 1.25 if n_con else 1)
        ax2.set_yticklabels([])

        fig.suptitle(f"Dissenter identity  [{lang.upper()}]  (n={n_samples} samples)",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        out = plots_dir / f"consensus_dissenters_{lang}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")

    # ── Figure 3: Who causes each split level? ───────────────────────────────
    if by_split_dis and len(by_split_dis) > 1:
        all_models = sorted({m for mv in by_split_dis.values() for m in mv})
        palette    = plt.cm.tab10.colors
        model_color = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}

        split_levels = list(by_split_dis.keys())
        x = range(len(split_levels))
        bar_w = 0.12
        n_models = len(all_models)
        offsets  = [(i - (n_models - 1) / 2) * bar_w for i in range(n_models)]

        fig, ax = plt.subplots(figsize=(max(6, len(split_levels) * 1.8), 4))
        for i, m in enumerate(all_models):
            vals = [by_split_dis[sl].get(m, 0) for sl in split_levels]
            bars = ax.bar([xi + offsets[i] for xi in x], vals,
                          width=bar_w, color=model_color[m],
                          label=_short(m), edgecolor="white", linewidth=0.3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.2,
                            str(v), ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(list(x))
        ax.set_xticklabels(split_levels, fontsize=10)
        ax.set_xlabel("Vote split level", fontsize=10)
        ax.set_ylabel("Times model was in minority", fontsize=10)
        ax.set_title(f"Who causes each split?  [{lang.upper()}]", fontsize=11, pad=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        fig.tight_layout()
        out = plots_dir / f"consensus_dissenter_by_split_{lang}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Per-language runner
# ---------------------------------------------------------------------------

_VALID_KINDS = {"individual", "majority_vote", "weighted_agg", "aggregator"}


def run_lang(
    lang: str,
    proposers: list[str],
    ignore_models: set[str],
) -> None:
    lang_dir   = RESULTS_ROOT / lang
    plots_dir  = lang_dir / "plots"
    csv_path   = lang_dir / f"metrics_summary_{lang}.csv"
    kappa_path = lang_dir / f"method_kappa_{lang}.json"

    if not csv_path.exists():
        print(f"[{lang}] metrics_summary_{lang}.csv not found — run pipeline first")
        return

    print(f"\n[{lang.upper()}] → {plots_dir}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Only rows with a known kind value
    csv_rows: list[dict[str, str]] = []
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("kind") in _VALID_KINDS:
                    csv_rows.append(dict(row))
    except Exception as e:
        print(f"  [WARN] Could not read CSV: {e}")

    # Sanity check: verify best_bacc model in CSV matches fresh computation from records
    judge_path = lang_dir / f"memerag_judgement_{lang}.json"
    if judge_path.exists():
        try:
            jrecords = [r for r in load_json(judge_path).get("records", []) if isinstance(r, dict)]
            active   = [m for m in proposers if m not in ignore_models]

            def _quick_bacc(model: str) -> float:
                pc = pt = nc = nt = 0
                for rec in jrecords:
                    gold = rec.get("gold_label")
                    out  = (rec.get("model_outputs") or {}).get(model)
                    lbl  = out.get("label") if isinstance(out, dict) else None
                    if   gold == "Supported":     pt += 1; pc += lbl == "Supported"
                    elif gold == "Not Supported": nt += 1; nc += lbl == "Not Supported"
                return (pc / pt + nc / nt) / 2 if pt and nt else 0.0

            fresh_best  = max(active, key=_quick_bacc).split("/")[-1] if active else None
            csv_bb_row  = next((r for r in csv_rows if r.get("name", "").startswith("llm_best_bacc(")), None)
            if fresh_best and csv_bb_row:
                csv_model = csv_bb_row["name"][len("llm_best_bacc("):-1]
                if csv_model != fresh_best:
                    print(f"  [WARN] best_bacc mismatch — CSV shows '{csv_model}' but fresh records pick '{fresh_best}'")
                    print(f"         Re-run pipeline to fix: python pipeline/pipeline.py --lang {lang}")
        except Exception:
            pass

    # Models = PROPOSERS from config, minus IGNORE_MODELS
    models = [m for m in proposers if m not in ignore_models]

    # Method pairwise kappa
    method_kappa: dict = {}
    if kappa_path.exists():
        try:
            method_kappa = load_json(kappa_path).get("method_pairwise_kappa", {})
        except Exception:
            pass

    # Kappa plots
    if models and method_kappa:
        plot_kappa_combined_ranked (lang, models, method_kappa, plots_dir)
        plot_kappa_combined_faceted(lang, models, method_kappa, plots_dir)
        plot_kappa_mds             (lang, models, method_kappa, plots_dir)
    else:
        what = ("models list" if not models else "") + \
               (" and " if not models and not method_kappa else "") + \
               (f"method_kappa_{lang}.json" if not method_kappa else "")
        print(f"  [SKIP] all kappa plots — missing {what}")

    if csv_rows:
        plot_bacc_grouped      (lang, csv_rows, plots_dir)
        plot_bacc_kappa_stacked(lang, csv_rows, plots_dir)
    else:
        print(f"  [SKIP] Balanced accuracy charts — no CSV rows found")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()

    parser = argparse.ArgumentParser(
        description="Visualize pipeline results (reads config.py, all flags optional)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--lang", nargs="+", choices=["en", "es", "de", "fr", "hi"],
        help=f"Language(s) to plot (config default: {cfg['lang']})",
    )
    group.add_argument("--all", action="store_true", help="Plot all languages")
    args = parser.parse_args()

    langs         = ["en", "es", "de", "fr", "hi"] if args.all \
                    else (args.lang or [cfg["lang"]])
    proposers     = cfg["proposers"]
    ignore_models = cfg["ignore_models"]

    for lang in langs:
        run_lang(lang, proposers, ignore_models)



if __name__ == "__main__":
    main()
