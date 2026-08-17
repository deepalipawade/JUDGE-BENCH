"""
Cross-language comparison plots — reads analysis_* JSON files.

Usage:
    python pipeline/visualize_crosslang.py
    python pipeline/visualize_crosslang.py --lang en de fr
    python pipeline/visualize_crosslang.py --all
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
PIPELINE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PIPELINE_DIR))
from utils.io import load_json


def _load_config() -> dict:
    spec = importlib.util.spec_from_file_location("config", PIPELINE_DIR / "config.py")
    cfg  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(cfg)                    # type: ignore[union-attr]
    return {
        "lang": getattr(cfg, "LANG", "en"),
    }


def _short(model: str) -> str:
    return model.split("/")[-1][:22]


def _save(fig, name: str) -> None:
    import matplotlib.pyplot as plt
    out_dir = RESULTS_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 1. Vote split distribution
# ---------------------------------------------------------------------------

def plot_vote_splits_crosslang(langs: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] vote splits — missing {e}")
        return

    lang_data: dict[str, dict] = {}
    for lang in langs:
        p = RESULTS_ROOT / lang / f"analysis_consensus_{lang}.json"
        if not p.exists():
            continue
        s = load_json(p).get("summary", {})
        splits = s.get("vote_split_distribution", {})
        if splits:
            lang_data[lang] = {
                "splits": splits,
                "n": s.get("n_samples", 0),
                "pct_unan": s.get("pct_unanimous", 0),
                "pct_majo_correct": s.get("pct_majority_correct", 0),
                "n_unan_wrong": s.get("unanimous_wrong", {}).get("n_cases", 0),
                "n_unan": s.get("n_unanimous", 0),
            }

    if not lang_data:
        print("  [SKIP] vote splits — no analysis_consensus files found")
        return

    SPLIT_COLORS = ["#4CAF50", "#8BC34A", "#FFC107", "#FF7043", "#E53935"]
    cols = 3
    rows = math.ceil(len(lang_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0),
                              constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, d) in enumerate(lang_data.items()):
        ax     = axes_flat[idx]
        splits = d["splits"]
        labels = list(splits.keys())
        counts = [splits[k] for k in labels]
        colors = [SPLIT_COLORS[min(i, len(SPLIT_COLORS) - 1)] for i in range(len(labels))]

        bars = ax.bar(labels, counts, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.65)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    str(cnt), ha="center", va="bottom", fontsize=9)

        n_uw = d["n_unan_wrong"]
        n_un  = d["n_unan"]
        if n_uw and counts:
            ax.text(0, counts[0] * 0.45,
                    f"{n_uw} wrong\n({n_uw/n_un*100:.0f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

        ax.set_title(
            f"{lang.upper()}\n"
            f"n={d['n']}  {d['pct_unan']:.0f}% unan  {d['pct_majo_correct']:.0f}% majo✓",
            fontsize=10, pad=6,
        )
        ax.set_ylim(0, max(counts) * 1.22 if counts else 1)
        ax.set_xlabel("Vote split", fontsize=9)
        ax.set_ylabel("# samples", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)

    for idx in range(len(lang_data), rows * cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Vote split distribution — all languages", fontsize=13, fontweight="bold")
    _save(fig, "vote_splits_crosslang.png")


# ---------------------------------------------------------------------------
# 2. Who causes each vote split
# ---------------------------------------------------------------------------

def plot_dissenter_by_split_crosslang(langs: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] dissenter-by-split — missing {e}")
        return

    lang_data: dict[str, dict] = {}
    all_models: list[str] = []
    for lang in langs:
        p = RESULTS_ROOT / lang / f"analysis_consensus_{lang}.json"
        if not p.exists():
            continue
        s   = load_json(p).get("summary", {})
        dbs = s.get("dissenter_counts", {}).get("by_split", {})
        if dbs:
            lang_data[lang] = dbs
            for split_counts in dbs.values():
                for m in split_counts:
                    if m not in all_models:
                        all_models.append(m)

    if not lang_data:
        print("  [SKIP] dissenter-by-split — no data found")
        return

    palette     = plt.cm.tab10.colors
    model_color = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}

    cols = 3
    rows = math.ceil(len(lang_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0),
                              constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    last_ax = None
    for idx, (lang, dbs) in enumerate(lang_data.items()):
        ax         = axes_flat[idx]
        split_keys = sorted(dbs.keys(), reverse=True)

        models_here = sorted(
            {m for sc in dbs.values() for m in sc},
            key=lambda m: sum(sc.get(m, 0) for sc in dbs.values()),
            reverse=True,
        )
        x     = list(range(len(split_keys)))
        width = 0.8 / max(len(models_here), 1)

        for mi, m in enumerate(models_here):
            counts  = [dbs[sk].get(m, 0) for sk in split_keys]
            offsets = [xi - 0.4 + (mi + 0.5) * width for xi in x]
            ax.bar(offsets, counts, width=width * 0.85,
                   color=model_color.get(m, "grey"), label=_short(m),
                   edgecolor="white", linewidth=0.4)

        ax.set_title(lang.upper(), fontsize=10, fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(split_keys, fontsize=9)
        ax.set_xlabel("Vote split", fontsize=9)
        ax.set_ylabel("# times in minority", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)
        last_ax = ax

    for idx in range(len(lang_data), rows * cols):
        axes_flat[idx].set_visible(False)

    if last_ax is not None:
        handles, labels = last_ax.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8.5, loc="lower right",
                   ncol=1, framealpha=0.9)

    fig.suptitle("Who causes each vote split — all languages", fontsize=13, fontweight="bold")
    _save(fig, "dissenter_by_split_crosslang.png")


# ---------------------------------------------------------------------------
# 3. Dissenter profile — who dissents and correct rate
# ---------------------------------------------------------------------------

def plot_dissenter_crosslang(langs: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"  [SKIP] dissenter profile — missing {e}")
        return

    lang_data: dict[str, dict] = {}
    all_models: list[str] = []
    for lang in langs:
        p = RESULTS_ROOT / lang / f"analysis_consensus_{lang}.json"
        if not p.exists():
            continue
        s   = load_json(p).get("summary", {})
        dis = s.get("dissenter_counts", {}).get("by_model", {})
        con = s.get("correct_contrarians", {}).get("by_model", {})
        if dis:
            lang_data[lang] = {"dis": dis, "con": con}
            for m in dis:
                if m not in all_models:
                    all_models.append(m)

    if not lang_data:
        print("  [SKIP] dissenter profile — no consensus data found")
        return

    palette     = plt.cm.tab10.colors
    model_color = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}

    cols = 3
    rows = math.ceil(len(lang_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.8))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, data) in enumerate(lang_data.items()):
        ax  = axes_flat[idx]
        dis = data["dis"]
        con = data["con"]

        models = [m for m in all_models if m in dis and dis[m].get("n_minority_votes", 0) > 0]
        models = sorted(models,
                        key=lambda m: con.get(m, {}).get("n_correct_vs_majority", 0)
                                      / dis[m]["n_minority_votes"],
                        reverse=True)

        short = [_short(m) for m in models]
        n_dis = [dis[m]["n_minority_votes"] for m in models]
        n_con = [con.get(m, {}).get("n_correct_vs_majority", 0) for m in models]
        y     = list(range(len(models)))

        for yi, (nd, m) in enumerate(zip(n_dis, models)):
            c = model_color[m]
            ax.barh(yi, nd, color=(*c[:3], 0.2), edgecolor=c,
                    linewidth=0.9, height=0.55)
        for yi, (nc, m) in enumerate(zip(n_con, models)):
            ax.barh(yi, nc, color=model_color[m], edgecolor="white",
                    linewidth=0.3, height=0.55)

        x_max = max(n_dis) if n_dis else 1
        for yi, (nd, nc) in enumerate(zip(n_dis, n_con)):
            rate = nc / nd * 100 if nd else 0
            ax.text(nd + x_max * 0.02, yi, f"{rate:.0f}%",
                    va="center", fontsize=7.5, color="#333")

        ax.set_yticks(y)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title(lang.upper(), fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("# samples", fontsize=8)
        ax.set_xlim(0, x_max * 1.22)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6)

    for idx in range(len(lang_data), rows * cols):
        axes_flat[idx].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor="grey", alpha=0.25, edgecolor="grey",
                       label="Total dissents (outline)"),
        mpatches.Patch(facecolor="grey",
                       label="Correct dissents   % = correct dissent rate"),
    ]
    fig.legend(handles=legend_handles, fontsize=9,
               loc="lower right", bbox_to_anchor=(0.98, 0.01), framealpha=0.9)
    fig.suptitle(
        "Dissenter profile — all languages\n"
        "Sorted by correct dissent rate (%) within each language",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "dissenter_crosslang.png")


# ---------------------------------------------------------------------------
# 4. Judge bias — FPR (lenient) vs FNR (strict) per model
# ---------------------------------------------------------------------------

def plot_judge_leniency_crosslang(langs: list[str]) -> None:
    """
    FPR = false positive rate = judge calls 'Supported' when truth is 'Not Supported' → lenient
    FNR = false negative rate = judge calls 'Not Supported' when truth is 'Supported' → strict
    ★ marks the model selected by best_bacc as aggregator.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"  [SKIP] judge bias — missing {e}")
        return

    lang_data: dict[str, dict] = {}
    all_models: list[str] = []
    for lang in langs:
        p = RESULTS_ROOT / lang / f"analysis_judge_bias_{lang}.json"
        if not p.exists():
            continue
        judges = load_json(p).get("judges", {})
        if not judges:
            continue
        feat_p   = RESULTS_ROOT / lang / f"analysis_agg_features_{lang}.json"
        best_sel = None
        if feat_p.exists():
            ms = load_json(feat_p).get("method_summary", {})
            best_sel = ms.get("best_bacc", {}).get("model_selected")
        lang_data[lang] = {"judges": judges, "best_bacc_model": best_sel}
        for m in judges:
            if m not in all_models:
                all_models.append(m)

    if not lang_data:
        print("  [SKIP] judge bias — no analysis_judge_bias files found")
        return

    cols = 3
    rows = math.ceil(len(lang_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.2))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    width = 0.35

    for idx, (lang, d) in enumerate(lang_data.items()):
        ax       = axes_flat[idx]
        judges   = d["judges"]
        best_sel = d["best_bacc_model"]

        # Sort by FPR desc — most lenient at top
        models = sorted(
            [m for m in all_models if m in judges],
            key=lambda m: judges[m].get("fpr", 0),
            reverse=True,
        )
        short = [_short(m) for m in models]
        fpr   = [judges[m].get("fpr", 0) for m in models]
        fnr   = [judges[m].get("fnr", 0) for m in models]
        y     = list(range(len(models)))

        # FPR bars (red-orange = lenient)
        ax.barh([yi - width / 2 for yi in y], fpr, height=width,
                color="#E57373", edgecolor="white", linewidth=0.4, label="FPR (lenient)")
        # FNR bars (blue = strict)
        ax.barh([yi + width / 2 for yi in y], fnr, height=width,
                color="#64B5F6", edgecolor="white", linewidth=0.4, label="FNR (strict)")

        # Value labels + ★ for best_bacc selected
        x_max = max(max(fpr, default=0), max(fnr, default=0))
        for yi, (fp, fn, m) in enumerate(zip(fpr, fnr, models)):
            star = " ★" if m == best_sel else ""
            ax.text(max(fp, fn) + x_max * 0.04, yi,
                    f"{fp:.2f} / {fn:.2f}{star}",
                    va="center", fontsize=7,
                    color="#B71C1C" if m == best_sel else "#333",
                    fontweight="bold" if m == best_sel else "normal")

        ax.set_yticks(y)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlim(0, x_max * 1.55 + 0.02)
        ax.set_title(lang.upper(), fontsize=10, fontweight="bold", pad=5)
        ax.set_xlabel("Error rate", fontsize=8.5)
        ax.invert_yaxis()
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6)

    for idx in range(len(lang_data), rows * cols):
        axes_flat[idx].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#E57373", label="FPR — calls 'Supported' when wrong  (lenient)"),
        mpatches.Patch(color="#64B5F6", label="FNR — calls 'Not Supported' when wrong  (strict)"),
    ]
    fig.legend(handles=legend_handles, fontsize=9, loc="lower center",
               ncol=1, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    fig.suptitle(
        "Judge error profile — all languages  (★ = best_bacc selected)\n"
        "FPR/FNR value shown as  fpr / fnr  per model",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "judge_leniency_crosslang.png")


# ---------------------------------------------------------------------------
# 5. Aggregator override majority — random vs best_bacc
# ---------------------------------------------------------------------------

def plot_agg_override_crosslang(langs: list[str]) -> None:
    """
    When the aggregator disagrees with majority vote, did it help or hurt?
      improved = agg correct, majority wrong  → aggregator adds value
      hurt     = majority correct, agg wrong  → aggregator degrades
      net gain = improved − hurt
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"  [SKIP] agg override — missing {e}")
        return

    FOCUS = ["random", "best_bacc"]   # only these two methods

    lang_data: dict[str, dict] = {}
    for lang in langs:
        p = RESULTS_ROOT / lang / f"analysis_agg_override_{lang}.json"
        if not p.exists():
            continue
        methods = load_json(p).get("methods", {})
        filtered = {k: v for k, v in methods.items() if k in FOCUS}
        if not filtered:
            continue
        feat_p = RESULTS_ROOT / lang / f"analysis_agg_features_{lang}.json"
        best_model = None
        if feat_p.exists():
            ms = load_json(feat_p).get("method_summary", {})
            best_model = ms.get("best_bacc", {}).get("model_selected")
        lang_data[lang] = {"methods": filtered, "best_model": best_model}

    if not lang_data:
        print("  [SKIP] agg override — no analysis_agg_override files found")
        return


    cols = 3
    rows = math.ceil(len(lang_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, entry) in enumerate(lang_data.items()):
        ax         = axes_flat[idx]
        methods    = entry["methods"]
        best_model = entry["best_model"]

        present  = [m for m in FOCUS if m in methods]
        n        = len(present)
        group_w  = 0.6
        bar_w    = group_w / 2

        # Collect global max for consistent y axis
        all_vals = [v for m in present for v in
                    [methods[m].get("override_improved", 0), methods[m].get("override_hurt", 0)]]
        ymax = max(all_vals) if all_vals else 1

        for gi, method in enumerate(present):
            d        = methods[method]
            improved = d.get("override_improved", 0)
            hurt     = d.get("override_hurt", 0)
            net      = d.get("override_net_gain", 0)
            cx       = gi  # centre of this method's group

            ax.bar(cx - bar_w / 2, improved, width=bar_w, color="#4CAF50",
                   edgecolor="white", linewidth=0.5)
            ax.bar(cx + bar_w / 2, hurt, width=bar_w, color="#EF5350",
                   edgecolor="white", linewidth=0.5)

            sign  = "+" if net > 0 else ("−" if net < 0 else "")
            badge = f"net {sign}{abs(net)}"
            ax.text(cx, ymax * 1.08, badge, ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#2E7D32" if net > 0 else ("#C62828" if net < 0 else "#555"),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.6))

        # Build x-axis tick labels: show best_bacc model name for the best_bacc bar
        def _xtick(m: str) -> str:
            if m == "best_bacc" and best_model:
                return f"best\nbacc\n({_short(best_model)})"
            return m.replace("_", "\n")

        ax.set_title(lang.upper(), fontsize=10, fontweight="bold", pad=5)
        ax.set_xticks(list(range(n)))
        ax.set_xticklabels([_xtick(m) for m in present], fontsize=8)
        ax.set_ylabel("# disagreements (agg ≠ majority)", fontsize=8.5)
        ax.set_ylim(0, ymax * 1.38)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)

    for idx in range(len(lang_data), rows * cols):
        axes_flat[idx].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#4CAF50", label="Improved  (agg correct, majority wrong)"),
        mpatches.Patch(color="#EF5350", label="Hurt  (majority correct, agg wrong)"),
    ]
    fig.legend(handles=legend_handles, fontsize=9, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    fig.suptitle(
        "When aggregator disagrees with majority vote — random vs best_bacc\n"
        "Green = aggregator correct  |  Red = aggregator wrong  |  net = improved − hurt",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "agg_override_crosslang.png")


# ---------------------------------------------------------------------------
# 6. Individual BAcc vs Agg BAcc (random selection) scatter
# ---------------------------------------------------------------------------

def plot_individual_vs_agg_bacc_crosslang(langs: list[str]) -> None:
    """
    Each point = one model in one language.
    X = individual BAcc (as a standalone judge).
    Y = BAcc when randomly selected as aggregator.
    Diagonal = y==x (no difference between roles).
    Above diagonal → better as aggregator than solo.
    ★ = model selected by best_bacc; its agg BAcc shown as horizontal dashed line.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] individual vs agg BAcc — missing {e}")
        return

    lang_data: dict[str, dict] = {}
    all_models: list[str] = []
    for lang in langs:
        p = RESULTS_ROOT / lang / f"analysis_agg_features_{lang}.json"
        if not p.exists():
            continue
        data     = load_json(p)
        per_model = data.get("per_model", {})
        ms        = data.get("method_summary", {})
        best_sel  = ms.get("best_bacc", {}).get("model_selected")
        best_bacc_overall = ms.get("best_bacc", {}).get("agg_bacc")
        random_overall    = ms.get("random",   {}).get("agg_bacc")

        points = {}
        for m, v in per_model.items():
            ind  = v.get("individual_bacc")
            agg  = v.get("agg_bacc_from_random")
            if ind is not None and agg is not None:
                points[m] = {"ind": ind, "agg": agg}
                if m not in all_models:
                    all_models.append(m)

        if points:
            lang_data[lang] = {
                "points": points,
                "best_sel": best_sel,
                "best_bacc_overall": best_bacc_overall,
                "random_overall": random_overall,
            }

    if not lang_data:
        print("  [SKIP] individual vs agg BAcc — no analysis_agg_features files found")
        return

    palette     = plt.cm.tab10.colors
    model_color = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}

    cols = 3
    rows = math.ceil(len(lang_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.4))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, d) in enumerate(lang_data.items()):
        ax       = axes_flat[idx]
        points   = d["points"]
        best_sel = d["best_sel"]
        best_overall = d["best_bacc_overall"]
        rand_overall = d["random_overall"]

        all_vals = [v["ind"] for v in points.values()] + [v["agg"] for v in points.values()]
        lo = min(all_vals) - 0.03
        hi = max(all_vals) + 0.03

        # Diagonal y = x
        ax.plot([lo, hi], [lo, hi], color="#ccc", linewidth=1.0,
                linestyle="--", zorder=1)

        # Horizontal lines for strategy-level BAcc
        if rand_overall is not None:
            ax.axhline(rand_overall, color="#90A4AE", linewidth=1.0,
                       linestyle=":", zorder=1, label=f"random BAcc={rand_overall:.3f}")
        if best_overall is not None:
            ax.axhline(best_overall, color="#66BB6A", linewidth=1.2,
                       linestyle="-.", zorder=1, label=f"best_bacc BAcc={best_overall:.3f}")

        # Scatter points
        for m, v in points.items():
            is_best = (m == best_sel)
            color   = model_color.get(m, "#888")
            ax.scatter(v["ind"], v["agg"],
                       color=color, s=90 if is_best else 55,
                       marker="*" if is_best else "o",
                       zorder=3, edgecolors="black" if is_best else "white",
                       linewidths=0.8)
            label = _short(m) + (" ★" if is_best else "")
            ax.annotate(label, (v["ind"], v["agg"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5,
                        fontweight="bold" if is_best else "normal",
                        color="#333")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(lang.upper(), fontsize=10, fontweight="bold", pad=5)
        ax.set_xlabel("Individual BAcc", fontsize=8.5)
        ax.set_ylabel("BAcc as aggregator (random)", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(linestyle="--", alpha=0.25, linewidth=0.6)

        # Inline legend for strategy lines
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85,
                  handlelength=1.5, borderpad=0.4)

    for idx in range(len(lang_data), rows * cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Individual BAcc vs BAcc as random aggregator — all languages\n"
        "Above diagonal → model performs better as aggregator than as solo judge  |  ★ = best_bacc pick",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "individual_vs_agg_bacc_crosslang.png")


# ---------------------------------------------------------------------------
# 7. Majority accuracy by vote split level
# ---------------------------------------------------------------------------

def plot_majority_accuracy_by_split_crosslang(langs: list[str]) -> None:
    # Requires: python pipeline/analyze.py --all  (generates analysis_vote_confidence_{lang}.json)
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] majority accuracy by split — missing {e}")
        return

    lang_data: dict[str, dict] = {}
    for lang in langs:
        p = RESULTS_ROOT / lang / f"analysis_vote_confidence_{lang}.json"
        if not p.exists():
            continue
        d = load_json(p).get("q1_majority_accuracy_by_split", {})
        if d:
            lang_data[lang] = d

    if not lang_data:
        print("  [SKIP] majority accuracy by split — run analyze.py --all first")
        return

    SPLIT_ORDER  = ["7-0", "6-1", "5-2", "4-3"]
    SPLIT_COLORS = ["#4CAF50", "#8BC34A", "#FFC107", "#EF5350"]

    cols = 3
    rows = math.ceil(len(lang_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    axes_flat  = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, splits) in enumerate(lang_data.items()):
        ax      = axes_flat[idx]
        keys    = [s for s in SPLIT_ORDER if s in splits]
        accs    = [splits[s]["accuracy_pct"] for s in keys]
        ns      = [splits[s]["n_samples"]    for s in keys]
        colors  = [SPLIT_COLORS[SPLIT_ORDER.index(s)] for s in keys]

        bars = ax.bar(keys, accs, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.6)
        for bar, acc, n in zip(bars, accs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{acc:.0f}%\n(n={n})",
                    ha="center", va="bottom", fontsize=8)

        ax.set_ylim(0, 115)
        ax.axhline(50, color="#bbb", linewidth=0.8, linestyle="--")
        ax.set_title(lang.upper(), fontsize=10, fontweight="bold", pad=5)
        ax.set_xlabel("Vote split", fontsize=9)
        ax.set_ylabel("Majority correct (%)", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)

    for idx in range(len(lang_data), rows * cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Majority vote accuracy drops as judge disagreement increases\n"
        "% of samples where majority vote == gold label, by split level",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "majority_accuracy_by_split_crosslang.png")


# ---------------------------------------------------------------------------
# 8. Pairwise kappa heatmap — mean + per language
# ---------------------------------------------------------------------------

def plot_kappa_heatmap_crosslang(langs: list[str]) -> None:
    """
    2×3 grid:
      [0,0] = mean kappa across all languages
      [0,1],[0,2],[1,0],[1,1],[1,2] = per-language kappa heatmaps (EN,DE,ES,FR,HI)
    Cell color = Cohen's kappa between judge pair. Diagonal = 1.0.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np  # noqa: F401
    except ImportError as e:
        print(f"  [SKIP] kappa heatmap — missing {e}")
        return

    p = RESULTS_ROOT / "analysis_crosslang_kappa.json"
    if not p.exists():
        print("  [SKIP] kappa heatmap — analysis_crosslang_kappa.json not found")
        return

    data  = load_json(p)
    pairs = data.get("judge_pairs", [])

    # Build ordered model list (by mean kappa desc, Apertus last)
    summary = data.get("per_judge_summary", {})
    models  = sorted(summary.keys(),
                     key=lambda m: summary[m].get("mean_kappa_all_pairs", 0),
                     reverse=True)
    short   = [_short(m) for m in models]
    n       = len(models)
    idx_of  = {m: i for i, m in enumerate(models)}

    def _build_matrix(key: str) -> list[list[float]]:
        mat = [[float("nan")] * n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 1.0
        for pair in pairs:
            a, b = pair["judge_a"], pair["judge_b"]
            if a not in idx_of or b not in idx_of:
                continue
            val = pair["per_lang"].get(key) if key != "mean" else pair.get("mean_kappa")
            if val is not None:
                i, j = idx_of[a], idx_of[b]
                mat[i][j] = val
                mat[j][i] = val
        return mat

    cmap   = plt.cm.YlOrRd
    norm   = mcolors.Normalize(vmin=0.0, vmax=1.0)
    panels = [("Mean\n(all langs)", "mean")] + [(lg.upper(), lg) for lg in langs]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = list(axes.flat)

    for ax_idx, (title, key) in enumerate(panels):
        ax  = axes_flat[ax_idx]
        mat = _build_matrix(key)

        im = ax.imshow([[v if not (v != v) else 0 for v in row] for row in mat],
                       cmap=cmap, norm=norm, aspect="auto")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = mat[i][j]
                if val != val:   # nan
                    continue
                text_color = "white" if val > 0.75 or val < 0.25 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5 if n > 6 else 8, color=text_color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(short, rotation=35, ha="right",
                           fontsize=7.5, fontweight="bold" if ax_idx == 0 else "normal")
        ax.set_yticks(range(n))
        ax.set_yticklabels(short, fontsize=7.5,
                           fontweight="bold" if ax_idx == 0 else "normal")
        ax.set_title(title, fontsize=10 if ax_idx == 0 else 9,
                     fontweight="bold", pad=5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    fig.suptitle(
        "Pairwise Cohen's kappa — judge agreement across languages\n"
        "Models ordered by mean kappa (most agreeable → least)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "kappa_heatmap_crosslang.png")


# ---------------------------------------------------------------------------
# 9. Co-dissent heatmap — which models form the 4-3 minority together?
# ---------------------------------------------------------------------------

def plot_codissent_heatmap_crosslang(langs: list[str]) -> None:
    """
    For every 4-3 vote split, 3 models are always in the minority together.
    This heatmap shows how often each pair of models co-appeared in that minority.

    Cell (i, j) = # 4-3 cases where both model i AND model j were in the minority.
    Diagonal    = total # 4-3 cases where that model was in the minority.
    High off-diagonal value → those two models form a recurring dissenting faction.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] co-dissent heatmap — missing {e}")
        return

    PROPOSERS = [
        "Qwen3.6-35B",
        "google/gemma-4-26b-a4b-it-maas",
        "MiniMax-M2.7",
        "GPT-OSS-120b",
        "meta/llama-3.3-70b-instruct-maas",
        "gemini-2.5-flash",
        "Apertus-8B",
    ]
    LABEL_MAP = {"Supported": 1, "Not Supported": 0}
    n = len(PROPOSERS)
    idx_of = {m: i for i, m in enumerate(PROPOSERS)}
    short  = [_short(m) for m in PROPOSERS]

    lang_mats: dict[str, list] = {}
    lang_n43:  dict[str, int]  = {}

    for lang in langs:
        p = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"
        if not p.exists():
            continue
        records = load_json(p).get("records", [])

        mat   = [[0] * n for _ in range(n)]
        n_43  = 0

        for rec in records:
            outs = rec.get("model_outputs", {})
            labels = {}
            for m in PROPOSERS:
                out = outs.get(m)
                if out:
                    lbl = LABEL_MAP.get(out.get("label"))
                    if lbl is not None:
                        labels[m] = lbl

            if len(labels) < 7:
                continue

            n_sup = sum(labels.values())
            n_not = 7 - n_sup

            # Only 4-3 splits (minority = the 3-model side)
            if n_sup != 3 and n_not != 3:
                continue

            minority_val  = 1 if n_sup == 3 else 0
            minority_mods = [m for m in PROPOSERS if labels.get(m) == minority_val]

            if len(minority_mods) != 3:
                continue

            n_43 += 1
            for ia_idx, ma in enumerate(minority_mods):
                for mb in minority_mods[ia_idx:]:
                    ia = idx_of[ma]
                    ib = idx_of[mb]
                    mat[ia][ib] += 1
                    if ia != ib:
                        mat[ib][ia] += 1

        if n_43 > 0:
            lang_mats[lang] = mat
            lang_n43[lang]  = n_43

    if not lang_mats:
        print("  [SKIP] co-dissent heatmap — no judgement files found")
        return

    cols = 3
    rows = math.ceil(len(lang_mats) / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4.5, rows * 4.2),
                             constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, mat) in enumerate(lang_mats.items()):
        ax = axes_flat[idx]

        # Off-diagonal max (diagonal is always higher; don't let it dominate colorscale)
        off_vals = [mat[i][j] for i in range(n) for j in range(n) if i != j]
        vmax = max(off_vals) if off_vals else 1

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

        for i in range(n):
            for j in range(n):
                val = mat[i][j]
                text_color = "white" if val > vmax * 0.65 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color=text_color,
                        fontweight="bold" if i == j else "normal")

        ax.set_xticks(range(n))
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(short, fontsize=7.5)
        ax.set_title(f"{lang.upper()}  ({lang_n43[lang]} 4-3 cases)",
                     fontsize=10, fontweight="bold", pad=5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    for idx in range(len(lang_mats), rows * cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "4-3 minority co-dissent — which models appear in the minority TOGETHER?\n"
        "Cell (i,j) = # cases where both i and j were in the 3-model minority side\n"
        "Diagonal = total 4-3 cases that model was in minority  |  high off-diagonal → recurring faction",
        fontsize=11, fontweight="bold",
    )
    _save(fig, "codissent_heatmap_crosslang.png")


# ---------------------------------------------------------------------------
# Error correlation helpers + 3 views
# ---------------------------------------------------------------------------

_PROPOSERS_EC = [
    "Qwen3.6-35B",
    "google/gemma-4-26b-a4b-it-maas",
    "MiniMax-M2.7",
    "GPT-OSS-120b",
    "meta/llama-3.3-70b-instruct-maas",
    "gemini-2.5-flash",
    "Apertus-8B",
]
_LABEL_MAP_EC = {"Supported": 1, "Not Supported": 0}


def _compute_error_matrices(langs: list[str]) -> dict[str, tuple]:
    """
    Returns {lang: (matrix, short_names, n_samples)}.
    matrix[i][j] = fraction of samples where both model_i AND model_j were wrong.
    Diagonal = individual error rate of model_i.
    """
    result = {}
    n = len(_PROPOSERS_EC)
    for lang in langs:
        p = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"
        if not p.exists():
            continue
        records   = load_json(p).get("records", [])
        co_error  = [[0] * n for _ in range(n)]
        total     = 0
        for rec in records:
            gold = _LABEL_MAP_EC.get(rec.get("gold_label"))
            if gold is None:
                continue
            outs  = rec.get("model_outputs", {})
            wrong = []
            valid = True
            for m in _PROPOSERS_EC:
                out = outs.get(m)
                if not out:
                    valid = False; break
                lbl = _LABEL_MAP_EC.get(out.get("label"))
                if lbl is None:
                    valid = False; break
                wrong.append(lbl != gold)
            if not valid:
                continue
            total += 1
            for i in range(n):
                for j in range(i, n):
                    if wrong[i] and wrong[j]:
                        co_error[i][j] += 1
                        if i != j:
                            co_error[j][i] += 1
        if total > 0:
            mat   = [[co_error[i][j] / total for j in range(n)] for i in range(n)]
            short = [_short(m) for m in _PROPOSERS_EC]
            result[lang] = (mat, short, total)
    return result


# 10b. Per-model average error overlap
def plot_error_overlap_per_model_crosslang(langs: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [SKIP] error overlap per model — {e}"); return

    matrices = _compute_error_matrices(langs)
    if not matrices:
        print("  [SKIP] error overlap per model — no data"); return

    n_models  = len(_PROPOSERS_EC)
    palette   = plt.cm.tab10.colors
    mod_color = {_short(m): palette[i % len(palette)]
                 for i, m in enumerate(_PROPOSERS_EC)}

    cols = 3
    rows = math.ceil(len(matrices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8),
                             constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, (mat, short, total)) in enumerate(matrices.items()):
        ax = axes_flat[idx]

        # Average overlap of each model with all others (off-diagonal mean)
        avgs = []
        for i in range(n_models):
            others = [mat[i][j] for j in range(n_models) if j != i]
            avgs.append(sum(others) / len(others))

        order  = sorted(range(n_models), key=lambda k: avgs[k], reverse=True)
        labels = [short[k] for k in order]
        values = [avgs[k]  for k in order]
        colors = [mod_color[short[k]] for k in order]

        y = list(range(n_models))
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.4, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean error overlap with others", fontsize=8.5)
        ax.set_title(f"{lang.upper()}  (n={total})", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6)
        ax.tick_params(axis="x", labelsize=8)
        ax.invert_yaxis()

        for yi, v in zip(y, values):
            ax.text(v + max(values) * 0.01, yi, f"{v:.3f}",
                    va="center", fontsize=7, color="#333")

    for idx in range(len(matrices), rows * cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Per-model mean error overlap — all languages\n"
        "High value = model's errors are shared with many others  |  "
        "Low value = model fails on unique samples",
        fontsize=12, fontweight="bold",
    )
    _save(fig, "error_overlap_per_model_crosslang.png")


# 10c. Dendrogram — cluster models by error similarity
def plot_error_overlap_dendrogram_crosslang(langs: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        import numpy as np
    except ImportError as e:
        print(f"  [SKIP] error dendrogram — {e}"); return

    matrices = _compute_error_matrices(langs)
    if not matrices:
        print("  [SKIP] error dendrogram — no data"); return

    cols = 3
    rows = math.ceil(len(matrices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0),
                             constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (lang, (mat, short, total)) in enumerate(matrices.items()):
        ax = axes_flat[idx]

        # Convert overlap (similarity) to distance: d = 1 - overlap / max_overlap
        np_mat = np.array(mat)
        # Use off-diagonal max as normaliser
        off_diag = np_mat.copy()
        np.fill_diagonal(off_diag, 0)
        max_val  = off_diag.max() or 1.0
        dist_mat = 1.0 - (off_diag / max_val)
        np.fill_diagonal(dist_mat, 0.0)

        condensed = squareform(dist_mat)
        Z = linkage(condensed, method="average")

        dendrogram(
            Z,
            labels=short,
            orientation="left",
            ax=ax,
            color_threshold=0,
            above_threshold_color="#4A90D9",
            leaf_font_size=8,
        )
        ax.set_title(f"{lang.upper()}  (n={total})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Distance (1 − normalised error overlap)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.spines[["top", "right", "left"]].set_visible(False)

    for idx in range(len(matrices), rows * cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Error similarity clustering — all languages\n"
        "Models that branch together share similar error patterns across samples",
        fontsize=12, fontweight="bold",
    )
    _save(fig, "error_overlap_dendrogram_crosslang.png")


# ---------------------------------------------------------------------------
# Method BAcc / Kappa comparison across languages
# ---------------------------------------------------------------------------

def plot_method_bacc_kappa_crosslang(langs: list[str]) -> None:
    """Grouped bar chart: BAcc (top) and Kappa (bottom) per aggregation method per language.

    Data source: algo_agg_{lang}.json → metrics section.
    Methods shown: majority, owi, isp, ds, mace (any subset present in the JSON).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"  [SKIP] method BAcc/Kappa — missing {e}")
        return

    METHOD_LABELS = {
        "majority": "Majority",
        "owi":      "OWI",
        "isp":      "ISP",
        "ds":       "Dawid-Skene",
        "mace":     "MACE",
    }
    METHOD_ORDER = ["majority", "owi", "isp", "ds", "mace"]
    COLORS = ["#90A4AE", "#64B5F6", "#4DB6AC", "#FF8A65", "#BA68C8"]

    # Collect data: {lang: {method: {bacc, kappa}}}
    lang_data: dict[str, dict[str, dict]] = {}
    for lang in langs:
        p = RESULTS_ROOT / lang / f"algo_agg_{lang}.json"
        if not p.exists():
            continue
        raw = load_json(p).get("metrics", {})
        if not raw:
            continue
        lang_data[lang] = {
            m: {"bacc": raw[m].get("balanced_accuracy"), "kappa": raw[m].get("cohen_kappa")}
            for m in METHOD_ORDER if m in raw
        }

    if not lang_data:
        print("  [SKIP] method BAcc/Kappa — no algo_agg files found")
        return

    present_methods = [m for m in METHOD_ORDER
                       if any(m in d for d in lang_data.values())]
    lang_list  = list(lang_data.keys())
    n_langs    = len(lang_list)
    n_methods  = len(present_methods)
    x          = np.arange(n_langs)
    bar_w      = 0.8 / n_methods

    fig, (ax_bacc, ax_kappa) = plt.subplots(2, 1, figsize=(max(7, n_langs * 1.6), 7),
                                             sharex=True)

    for mi, method in enumerate(present_methods):
        offset = (mi - n_methods / 2 + 0.5) * bar_w
        color  = COLORS[mi % len(COLORS)]
        label  = METHOD_LABELS.get(method, method)

        baccs  = [lang_data[lg].get(method, {}).get("bacc") for lg in lang_list]
        kappas = [lang_data[lg].get(method, {}).get("kappa") for lg in lang_list]

        for ax, vals in ((ax_bacc, baccs), (ax_kappa, kappas)):
            for i, v in enumerate(vals):
                if v is None:
                    continue
                # pct string ("84.17%") or float
                num = float(str(v).rstrip("%")) if v is not None else None
                bar = ax.bar(x[i] + offset, num, bar_w * 0.9,
                             color=color, label=label if i == 0 else None,
                             edgecolor="white", linewidth=0.4)
                ax.text(x[i] + offset, num + 0.3, f"{num:.1f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)

    for ax, metric in ((ax_bacc, "Balanced Accuracy (%)"), (ax_kappa, "Cohen's Kappa (%)")):
        ax.set_ylabel(metric, fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([lg.upper() for lg in lang_list], fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ax_bacc.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[i])
                 for i, m in enumerate(present_methods)],
        labels=[METHOD_LABELS.get(m, m) for m in present_methods],
        fontsize=8, ncol=n_methods, loc="upper right",
        framealpha=0.85, handlelength=1.2,
    )

    fig.suptitle(
        "Aggregation method comparison across languages\n"
        "BAcc and Kappa for Majority, OWI, ISP, Dawid-Skene, MACE",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "method_bacc_kappa_crosslang.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()

    parser = argparse.ArgumentParser(
        description="Generate cross-language comparison plots from analysis_* JSON files."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--lang", nargs="+", choices=["en", "es", "de", "fr", "hi"],
        help=f"Languages to include (default: {cfg['lang']})",
    )
    group.add_argument("--all", action="store_true", help="Use all 5 languages")
    args = parser.parse_args()

    langs = ["en", "es", "de", "fr", "hi"] if args.all else (args.lang or [cfg["lang"]])

    print(f"[CROSS-LANG] languages: {langs}")
    plot_vote_splits_crosslang(langs)
    plot_dissenter_by_split_crosslang(langs)
    plot_codissent_heatmap_crosslang(langs)
    plot_dissenter_crosslang(langs)
    plot_judge_leniency_crosslang(langs)
    plot_agg_override_crosslang(langs)
    plot_individual_vs_agg_bacc_crosslang(langs)
    plot_majority_accuracy_by_split_crosslang(langs)
    plot_kappa_heatmap_crosslang(langs)
    plot_error_overlap_per_model_crosslang(langs)
    plot_error_overlap_dendrogram_crosslang(langs)
    plot_method_bacc_kappa_crosslang(langs)


if __name__ == "__main__":
    main()
