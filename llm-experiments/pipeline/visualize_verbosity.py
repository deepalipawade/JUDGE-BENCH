"""
Verbosity analysis: mean response length per judge vs balanced accuracy against gold.

Reads model outputs from algo_agg_{lang}.json and plots a scatter chart with
annotated model names.

Usage:
    python pipeline/visualize_verbosity.py
    python pipeline/visualize_verbosity.py --lang de
    python pipeline/visualize_verbosity.py --lang en de --unit words
    python pipeline/visualize_verbosity.py --all
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
PIPELINE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PIPELINE_DIR))
from utils.io import load_json

VALID_LABELS = {"Supported", "Not Supported"}


def _load_config() -> dict:
    spec = importlib.util.spec_from_file_location("config", PIPELINE_DIR / "config.py")
    cfg  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(cfg)                    # type: ignore[union-attr]
    return {
        "lang":         getattr(cfg, "LANG", "en"),
        "ignore_models": getattr(cfg, "IGNORE_MODELS", set()),
    }


def _verbosity(records: list[dict], model: str, unit: str) -> float | None:
    """Mean response length for a model across all records."""
    lengths = []
    for rec in records:
        out = rec.get("model_outputs", {}).get(model) or {}
        text = out.get("raw") or out.get("reason") or ""
        if not text:
            continue
        lengths.append(len(text.split()) if unit == "words" else len(text))
    return sum(lengths) / len(lengths) if lengths else None


def _bacc(records: list[dict], model: str) -> float | None:
    """Balanced accuracy of a model against gold_label."""
    pos_total = pos_correct = neg_total = neg_correct = 0
    for rec in records:
        gold = rec.get("gold_label")
        out  = rec.get("model_outputs", {}).get(model) or {}
        pred = out.get("label")
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
    return 0.5 * (pos_correct / pos_total + neg_correct / neg_total) * 100


def _short(model: str) -> str:
    return model.split("/")[-1]


def plot_verbosity_vs_bacc(
    lang: str,
    records: list[dict],
    models: list[str],
    unit: str,
    plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping verbosity plot")
        return

    xs, ys, labels = [], [], []
    for model in models:
        v = _verbosity(records, model, unit)
        b = _bacc(records, model)
        if v is not None and b is not None:
            xs.append(v)
            ys.append(b)
            labels.append(_short(model))

    if not xs:
        print(f"  [SKIP] No verbosity data for {lang}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=80, color="#4C9BE8", edgecolors="#1a5a8a", linewidth=0.8, zorder=3)

    for x, y, lbl in zip(xs, ys, labels):
        txt = ax.text(x, y + 0.4, lbl, fontsize=8, ha="center", va="bottom")
        txt.set_path_effects([
            pe.withStroke(linewidth=2, foreground="white"),
        ])

    unit_label = "words" if unit == "words" else "characters"
    ax.set_xlabel(f"Mean response length ({unit_label})", fontsize=10)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=10)
    ax.set_title(f"Model verbosity vs accuracy  [{lang.upper()}]", fontsize=12, pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(50.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(max(xs), 50.5, "chance", color="grey", fontsize=7, ha="right", va="bottom")

    # Correlation note
    if len(xs) > 2:
        import statistics
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        sd_x = statistics.stdev(xs)
        sd_y = statistics.stdev(ys)
        if sd_x > 0 and sd_y > 0:
            r = cov / ((len(xs) - 1) * sd_x * sd_y)
            ax.text(0.97, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=9, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))

    fig.tight_layout()
    out = plots_dir / f"verbosity_vs_bacc_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def run_lang(lang: str, unit: str, ignore_models: set[str]) -> None:
    lang_dir  = RESULTS_ROOT / lang
    plots_dir = lang_dir / "plots"
    algo_path = lang_dir / f"algo_agg_{lang}.json"

    if not algo_path.exists():
        print(f"[{lang}] algo_agg_{lang}.json not found — run pipeline first")
        return

    print(f"\n[{lang.upper()}] → {plots_dir}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    data    = load_json(algo_path)
    models  = [m for m in data.get("models", []) if m not in ignore_models]
    records = data.get("records", [])

    if not models or not records:
        print(f"  [SKIP] No models or records found in algo_agg_{lang}.json")
        return

    plot_verbosity_vs_bacc(lang, records, models, unit, plots_dir)


def main() -> None:
    cfg = _load_config()

    parser = argparse.ArgumentParser(
        description="Plot model verbosity vs balanced accuracy."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lang", nargs="+", choices=["en", "es", "de", "fr", "hi"],
                       help=f"Language(s) to plot (config default: {cfg['lang']})")
    group.add_argument("--all", action="store_true", help="Run for all languages")
    parser.add_argument("--unit", choices=["words", "chars"], default="words",
                        help="Verbosity unit (default: words)")
    args = parser.parse_args()

    langs = ["en", "es", "de", "fr", "hi"] if args.all else (args.lang or [cfg["lang"]])
    for lang in langs:
        run_lang(lang, args.unit, cfg["ignore_models"])


if __name__ == "__main__":
    main()
