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
        "lang":           getattr(cfg, "LANG", "en"),
        "aggregator_llm": getattr(cfg, "AGGREGATOR_LLM", None),
        "seed":           getattr(cfg, "SEED", 42),
        "proposers":      set(getattr(cfg, "PROPOSERS", [])),
        "ignore_models":  set(getattr(cfg, "IGNORE_MODELS", set())),
    }


def _combined_method_name(aggregator_llm: str | None, seed: int) -> str | None:
    if aggregator_llm is None:
        return None
    if aggregator_llm == "random":
        return f"random_seed{seed}"
    return aggregator_llm


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
# Kappa plot 6 — Ranked horizontal bar chart
# ---------------------------------------------------------------------------

def plot_kappa_ranked_bar(
    lang: str, models: list[str], method_kappa: dict, plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError as e:
        print(f"  [SKIP] ranked bar — missing {e}")
        return

    mat   = _kappa_matrix(models, method_kappa)
    names = [_short(m) for m in models]
    n     = len(models)

    pairs: list[tuple[str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            label = f"{names[i]}  vs  {names[j]}"
            pairs.append((label, mat[i][j]))

    pairs.sort(key=lambda x: x[1])
    labels, kappas = zip(*pairs)

    cmap   = matplotlib.colormaps["RdYlGn"]
    colors = [cmap(k) for k in kappas]

    fig, ax = plt.subplots(figsize=(8, max(4, len(pairs) * 0.45)))
    bars = ax.barh(range(len(pairs)), kappas, color=colors, edgecolor="white",
                   linewidth=0.4)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cohen's Kappa")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.6, color="grey", linestyle="--", linewidth=0.7, alpha=0.5,
               label="κ = 0.6")
    ax.legend(fontsize=8)

    for bar, k in zip(bars, kappas):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{k:.2f}", va="center", fontsize=7)

    ax.set_title(f"Inter-judge Kappa — All pairs ranked  [{lang.upper()}]",
                 fontsize=12, pad=10)
    fig.tight_layout()
    out = plots_dir / f"interjudge_kappa_ranked_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Kappa plot 7 — Faceted bar chart (one subplot per model)
# ---------------------------------------------------------------------------

def plot_kappa_faceted(
    lang: str, models: list[str], method_kappa: dict, plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] faceted — missing {e}")
        return

    mat   = _kappa_matrix(models, method_kappa)
    names = [_short(m) for m in models]
    n     = len(models)
    cols  = min(n, 4)
    rows  = math.ceil(n / cols)

    # One consistent color per model (same color for a model across all subplots)
    palette = plt.cm.tab10.colors
    model_color = {names[i]: palette[i % len(palette)] for i in range(n)}

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3.5, rows * 3),
                              sharey=False)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, model in enumerate(models):
        ax = axes_list[idx]
        others  = [j for j in range(n) if j != idx]
        o_names = [names[j] for j in others]
        o_kaps  = [mat[idx][j] for j in others]
        # sort descending
        paired  = sorted(zip(o_kaps, o_names), reverse=True)
        o_kaps, o_names = zip(*paired) if paired else ([], [])

        colors = [model_color[nm] for nm in o_names]
        ax.barh(range(len(o_names)), o_kaps, color=colors, edgecolor="white",
                linewidth=0.4)
        ax.set_yticks(range(len(o_names)))
        ax.set_yticklabels(o_names, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_title(names[idx], fontsize=9, fontweight="bold")
        ax.axvline(0.6, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("κ", fontsize=7)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes_list[idx].set_visible(False)

    fig.suptitle(f"Inter-judge Kappa — Per-model view  [{lang.upper()}]",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = plots_dir / f"interjudge_kappa_faceted_{lang}.png"
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
        import numpy as np
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

    import matplotlib
    cmap   = matplotlib.colormaps["RdYlGn"]
    colors = [cmap(k) for k in mean_kappas]

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
        "weighted_agg":  None,       # sub-colors below
        "aggregator":    "#D0021B",
    }
    ALGO_COLORS = ["#2d8a4e", "#5BAD6F", "#a8d8b9"]   # OWI, ISP, DS shades
    AGG_COLORS  = ["#B00000", "#D0021B", "#e85454"]   # LLM aggregator shades
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

    # Build x positions with a larger gap between groups
    GROUP_GAP    = 1.2
    CLUSTER_KINDS = {"weighted_agg", "aggregator"}
    ALGO_WIDTH   = 0.3   # tighter spacing inside cluster groups
    bar_x, bar_h, bar_c, bar_lbl = [], [], [], []
    group_centers: dict[str, float] = {}

    x = 0.0
    for gidx, kind in enumerate(GROUPS):
        items = by_kind[kind]
        if not items:
            continue
        if bar_x:            # gap before each group
            x += GROUP_GAP
        x_start = x
        for i, (name, val) in enumerate(items):
            spacing = ALGO_WIDTH if kind in CLUSTER_KINDS else 1.0
            bar_x.append(x)
            bar_h.append(val)
            bar_lbl.append(name)
            if kind == "weighted_agg":
                bar_c.append(ALGO_COLORS[i % len(ALGO_COLORS)])
            elif kind == "aggregator":
                bar_c.append(AGG_COLORS[i % len(AGG_COLORS)])
            else:
                bar_c.append(KIND_COLOR[kind])
            x += spacing
        group_centers[kind] = (x_start + x - spacing) / 2

    if not bar_x:
        print(f"  [SKIP] bacc grouped — no data")
        return

    fig, ax = plt.subplots(figsize=(max(9, len(bar_x) * 0.8 + 2), 5))
    bars = ax.bar(bar_x, bar_h, color=bar_c, width=0.7,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(50.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Group background shading
    KIND_BG = {
        "individual": "#e8f0fa",
        "majority_vote": "#fef4e0",
        "weighted_agg": "#e8f6ed",
        "aggregator": "#fce8e8",
    }
    x = 0.0
    for kind in GROUPS:
        items = by_kind[kind]
        if not items:
            continue
        spacing = ALGO_WIDTH if kind in CLUSTER_KINDS else 1.0
        n_items = len(items)
        span = (n_items - 1) * spacing
        x_lo = x - 0.45
        x_hi = x + span + 0.45
        ax.axvspan(x_lo, x_hi, alpha=0.25,
                   color=KIND_BG.get(kind, "#eee"), zorder=0)
        x += span + (GROUP_GAP if kind != GROUPS[-1] else 0) + 1.0

    # Value labels
    for bar, val in zip(bars, bar_h):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

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
        mpatches.Patch(color="#D0021B", label="LLM aggregator"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = plots_dir / f"balanced_accuracy_grouped_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")




# ---------------------------------------------------------------------------
# Per-language runner
# ---------------------------------------------------------------------------

_VALID_KINDS = {"individual", "majority_vote", "weighted_agg", "aggregator"}


def run_lang(
    lang: str,
    aggregator_llm: str | None,
    seed: int,
    proposers: set[str],
    ignore_models: set[str],
) -> None:
    lang_dir   = RESULTS_ROOT / lang
    plots_dir  = lang_dir / "plots"
    csv_path   = lang_dir / f"metrics_summary_{lang}.csv"
    kappa_path = lang_dir / f"method_kappa_{lang}.json"
    algo_path  = lang_dir / f"algo_agg_{lang}.json"

    method_name   = _combined_method_name(aggregator_llm, seed)
    combined_path = (lang_dir / f"algo_agg_{method_name}_{lang}.json"
                     if method_name else None)

    if not csv_path.exists():
        print(f"[{lang}] metrics_summary_{lang}.csv not found — run pipeline first")
        return

    print(f"\n[{lang.upper()}] → {plots_dir}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Section 1 — only rows with a known kind value
    csv_rows: list[dict[str, str]] = []
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("kind") in _VALID_KINDS:
                    csv_rows.append(dict(row))
    except Exception as e:
        print(f"  [WARN] Could not read CSV: {e}")

    # Model list from saved JSON, then filter to current config
    raw_models: list[str] = []
    for path in [algo_path, combined_path]:
        if path and path.exists():
            try:
                raw_models = load_json(path).get("models", [])
                if raw_models:
                    break
            except Exception:
                pass

    # Keep only models in current PROPOSERS (if set) and not in IGNORE_MODELS
    if proposers:
        models = [m for m in raw_models if m in proposers and m not in ignore_models]
    else:
        models = [m for m in raw_models if m not in ignore_models]

    if raw_models and len(models) < len(raw_models):
        removed = set(raw_models) - set(models)
        print(f"  [config filter] removed from plots: {', '.join(sorted(removed))}")

    # Method pairwise kappa
    method_kappa: dict = {}
    if kappa_path.exists():
        try:
            method_kappa = load_json(kappa_path).get("method_pairwise_kappa", {})
        except Exception:
            pass

    # Kappa plots
    if models and method_kappa:
        plot_kappa_ranked_bar(lang, models, method_kappa, plots_dir)
        plot_kappa_faceted   (lang, models, method_kappa, plots_dir)
        plot_kappa_mds       (lang, models, method_kappa, plots_dir)
    else:
        what = ("models list" if not models else "") + \
               (" and " if not models and not method_kappa else "") + \
               (f"method_kappa_{lang}.json" if not method_kappa else "")
        print(f"  [SKIP] all kappa plots — missing {what}")

    if csv_rows:
        plot_bacc_grouped(lang, csv_rows, plots_dir)
    else:
        print(f"  [SKIP] Balanced accuracy chart — no CSV rows found")


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
    parser.add_argument(
        "--aggregator_llm",
        choices=["random", "best_bacc", "ds_rank", "fixed"],
        default=None,
        help=f"Aggregator method for combined file lookup (config default: {cfg['aggregator_llm']})",
    )
    args = parser.parse_args()

    langs          = ["en", "es", "de", "fr", "hi"] if args.all \
                     else (args.lang or [cfg["lang"]])
    aggregator_llm = args.aggregator_llm or cfg["aggregator_llm"]
    seed           = cfg["seed"]
    proposers      = cfg["proposers"]
    ignore_models  = cfg["ignore_models"]

    for lang in langs:
        run_lang(lang, aggregator_llm, seed, proposers, ignore_models)


if __name__ == "__main__":
    main()
