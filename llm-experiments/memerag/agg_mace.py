"""
Standalone MACE analysis script.

Reads votes from results_tmp/memerag_ext/votes/votes_{lang}.csv and runs MACE
independently of the main pipeline. Always produces two files — one for VB
inference and one for EM inference.

Usage:
    # Single language, default hyperparams (saves mace_vb_single_run + mace_em_single_run)
    python llm-experiments/memerag/agg_mace.py --lang en

    # Sweep hyperparams (saves mace_vb_sweep + mace_em_sweep)
    python llm-experiments/memerag/agg_mace.py --lang en --sweep

    # All languages, default hyperparams
    python llm-experiments/memerag/agg_mace.py --all

    # All languages + sweep
    python llm-experiments/memerag/agg_mace.py --all --sweep

    # Semi-supervised sweep (vary oracle control fraction)
    python llm-experiments/memerag/agg_mace.py --lang en --control-sweep
    python llm-experiments/memerag/agg_mace.py --all --control-sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from utils.mace import run_mace

ROOT         = Path(__file__).resolve().parents[2]
VOTES_DIR    = ROOT / "results_tmp" / "memerag_ext" / "votes"
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
LANGS        = ["en", "de", "es", "fr", "hi"]

# Representative alpha/beta pairs covering qualitatively distinct prior regions:
#   symmetric (α=β): equal weight on spamming vs knowing
#   reliable bias (α<β): prior that annotators know the answer
#   unreliable bias (α>β): prior that annotators are guessing
ALPHA_BETA_PAIRS: list[tuple[float, float]] = [
    (0.1, 0.1),   # symmetric, diffuse (weak prior)
    (0.5, 0.5),   # Jeffreys prior — default, uninformative
    (2.0, 2.0),   # symmetric, concentrated (strong prior)
    (0.1, 1.0),   # reliable bias, moderate
    (0.5, 2.0),   # reliable bias, stronger
    (1.0, 0.1),   # unreliable bias, moderate
    (2.0, 0.5),   # unreliable bias, stronger
]

# VB sweep: n_restarts × n_iter × ALPHA_BETA_PAIRS = 2×2×7 = 28 configs
SWEEP_GRID_VB = {
    "n_restarts": [5, 20],
    "n_iter":     [50, 100],
}

# EM sweep: smoothing only (no alpha/beta — those are VB priors)
# Smoothing default per MACE paper = 0.01 / num_labels = 0.005 for binary
SWEEP_GRID_EM = {
    "n_restarts": [5, 20],
    "n_iter":     [50, 100],
    "smoothing":  [0.001, 0.005, 0.01],
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_votes(lang: str) -> tuple[list[str], list[int | None], dict[str, list[int | None]], list[str]]:
    """Returns (sample_ids, gold_labels, matrix, model_names)."""
    path = VOTES_DIR / f"votes_{lang}.csv"
    if not path.exists():
        print(f"[ERROR] {path} not found. Run pipeline --no-llm-agg first.")
        sys.exit(1)

    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    model_cols = [c for c in reader.fieldnames if c not in ("sample_id", "gold_label")]

    sample_ids: list[str]               = []
    gold_labels: list[int | None]       = []
    matrix: dict[str, list[int | None]] = {m: [] for m in model_cols}

    for row in rows:
        sample_ids.append(row["sample_id"])
        g = row["gold_label"]
        gold_labels.append(int(g) if g != "" else None)
        for m in model_cols:
            v = row[m]
            matrix[m].append(int(v) if v != "" else None)

    return sample_ids, gold_labels, matrix, model_cols


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: list[int | None],
    gold: list[int | None],
) -> dict[str, float | None]:
    tp = fp = tn = fn = 0
    for p, g in zip(preds, gold):
        if p is None or g is None:
            continue
        if g == 1 and p == 1:   tp += 1
        elif g == 0 and p == 1: fp += 1
        elif g == 0 and p == 0: tn += 1
        elif g == 1 and p == 0: fn += 1

    tpr  = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr  = tn / (tn + fp) if (tn + fp) > 0 else None
    bacc = (tpr + tnr) / 2 if (tpr is not None and tnr is not None) else None

    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec  = tpr
    f1   = 2 * prec * rec / (prec + rec) if (prec and rec) else None

    n  = tp + fp + tn + fn
    po = (tp + tn) / n if n > 0 else None
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n) if n > 0 else None
    kappa = (po - pe) / (1 - pe) if (po is not None and pe is not None and pe != 1) else None

    return {
        "bacc":      bacc,
        "kappa":     kappa,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "tpr":       tpr,
        "tnr":       tnr,
        "n_valid":   n,
    }


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "  n/a"
    return f"{v * 100:6.2f}%" if pct else f"{v:.4f}"


# ---------------------------------------------------------------------------
# Control items (semi-supervised)
# ---------------------------------------------------------------------------

ORACLE_KEY = "_oracle_control"

def add_control_annotator(
    matrix: dict[str, list[int | None]],
    gold: list[int | None],
    frac: float,
    seed: int = 42,
) -> dict[str, list[int | None]]:
    """Inject a synthetic oracle annotator that votes correctly for `frac` of items.

    Items not selected as controls get None (abstain). MACE infers this annotator
    as highly competent, anchoring competence estimates for real judges.
    Stratified sampling ensures equal proportion of each gold label is controlled.
    """
    import random
    rng = random.Random(seed)

    pos_idx = [i for i, g in enumerate(gold) if g == 1]
    neg_idx = [i for i, g in enumerate(gold) if g == 0]

    n_pos = max(1, round(len(pos_idx) * frac))
    n_neg = max(1, round(len(neg_idx) * frac))
    control_idx = set(rng.sample(pos_idx, min(n_pos, len(pos_idx))) +
                      rng.sample(neg_idx, min(n_neg, len(neg_idx))))

    oracle_votes = [gold[i] if i in control_idx else None for i in range(len(gold))]
    return {**matrix, ORACLE_KEY: oracle_votes}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def apply_threshold(
    labels: list[int | None],
    entropy: list[float],
    threshold: float,
) -> list[int | None]:
    """Set predictions to None for samples above the entropy threshold percentile.

    threshold=1.0 keeps all predictions (no filtering).
    threshold=0.8 nulls out the 20% most uncertain samples.
    """
    if threshold >= 1.0:
        return labels
    valid = [(i, e) for i, e in enumerate(entropy) if not math.isnan(e)]
    if not valid:
        return labels
    valid.sort(key=lambda x: x[1])
    keep_n   = max(1, int(len(valid) * threshold))
    keep_idx = {i for i, _ in valid[:keep_n]}
    return [l if i in keep_idx else None for i, l in enumerate(labels)]


def run_single(
    lang: str,
    method: str,
    n_restarts: int = 10,
    n_iter: int = 50,
    smoothing: float = 0.005,   # MACE paper default = 0.01 / num_labels = 0.005 for binary
    default_noise: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    threshold: float = 1.0,     # 1.0 = keep all; <1.0 = drop highest-entropy samples
    control_frac: float = 0.0,  # 0.0 = unsupervised; >0 = semi-supervised with oracle
    seed: int = 42,
    sample_ids: list[str] | None = None,
    gold: list[int | None] | None = None,
    matrix: dict[str, list[int | None]] | None = None,
    models: list[str] | None = None,
) -> dict:
    """Run MACE with a single config and save mace_{method}_single_run_{lang}.json."""
    if sample_ids is None:
        sample_ids, gold, matrix, models = load_votes(lang)

    run_matrix = matrix
    run_models = models
    if control_frac > 0.0:
        run_matrix = add_control_annotator(matrix, gold, frac=control_frac, seed=seed)
        run_models = models + [ORACLE_KEY]

    labels, competence, entropy = run_mace(
        run_models, run_matrix,
        n_restarts=n_restarts,
        n_iter=n_iter,
        method=method,
        smoothing=smoothing,
        default_noise=default_noise,
        alpha=alpha,
        beta=beta,
        random_state=seed,
    )

    labels     = apply_threshold(labels, entropy, threshold)
    metrics    = compute_metrics(labels, gold)
    competence = {m: c for m, c in competence.items() if m != ORACLE_KEY}

    print(f"\n{'='*60}")
    print(f"  MACE-{method.upper()}  |  lang={lang.upper()}", end="")
    if control_frac > 0.0:
        print(f"  control={control_frac:.0%}", end="")
    print()
    if method == "vb":
        print(f"  n_restarts={n_restarts}  n_iter={n_iter}  alpha={alpha}  beta={beta}")
    else:
        print(f"  n_restarts={n_restarts}  n_iter={n_iter}  smoothing={smoothing}")
    print(f"{'='*60}")

    print("\n── Prediction metrics (vs gold label) ──")
    print(f"  Balanced Accuracy : {fmt(metrics['bacc'])}")
    print(f"  Cohen's Kappa     : {fmt(metrics['kappa'])}")
    print(f"  Precision         : {fmt(metrics['precision'])}")
    print(f"  Recall (TPR)      : {fmt(metrics['recall'])}")
    print(f"  F1 Score          : {fmt(metrics['f1'])}")
    print(f"  TNR (specificity) : {fmt(metrics['tnr'])}")
    print(f"  Valid samples     : {int(metrics['n_valid'])}/{len(labels)}")

    print("\n── Per-model competence (non-spamming probability) ──")
    print(f"  {'Model':<44} {'Competence':>10}")
    print(f"  {'-'*56}")
    for m, c in sorted(competence.items(), key=lambda x: -x[1]):
        bar = "█" * int(c * 20)
        print(f"  {m:<44} {c:>10.4f}  {bar}")

    valid_entropy = [e for e in entropy if not math.isnan(e)]
    if valid_entropy:
        mean_h = sum(valid_entropy) / len(valid_entropy)
        max_h  = max(valid_entropy)
        min_h  = min(valid_entropy)
        high_n = sum(1 for e in valid_entropy if e > 0.5)
        print(f"\n── Per-sample entropy (uncertainty) ──")
        print(f"  Mean entropy       : {mean_h:.4f}")
        print(f"  Min / Max          : {min_h:.4f} / {max_h:.4f}")
        print(f"  High-entropy (>0.5): {high_n}/{len(valid_entropy)} samples")

        if high_n > 0:
            uncertain = sorted(
                [(i, e) for i, e in enumerate(entropy) if not math.isnan(e)],
                key=lambda x: -x[1],
            )[:5]
            print(f"\n  Top-5 most uncertain samples:")
            print(f"  {'sample_id':<15} {'gold':>6} {'pred':>6} {'entropy':>10}")
            print(f"  {'-'*42}")
            for i, e in uncertain:
                g = str(gold[i]) if gold[i] is not None else "?"
                p = str(labels[i]) if labels[i] is not None else "?"
                print(f"  {sample_ids[i]:<15} {g:>6} {p:>6} {e:>10.4f}")

    n_sup  = sum(1 for l in labels if l == 1)
    n_not  = sum(1 for l in labels if l == 0)
    n_miss = sum(1 for l in labels if l is None)
    print(f"\n── Predicted label distribution ──")
    print(f"  Supported     : {n_sup}")
    print(f"  Not Supported : {n_not}")
    print(f"  Missing       : {n_miss}")

    # Config dict stored in JSON body (not in filename)
    config = {
        "method":        method,
        "n_restarts":    n_restarts,
        "n_iter":        n_iter,
        "default_noise": default_noise,
        "threshold":     threshold,
        "control_frac":  control_frac,
        "seed":          seed,
    }
    if method == "vb":
        config["alpha"] = alpha
        config["beta"]  = beta
    else:
        config["smoothing"] = smoothing  # smoothing is EM-only

    out_dir = RESULTS_ROOT / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mace_{method}_single_run_{lang}.json"
    out_path.write_text(json.dumps({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "lang":       lang,
        "config":     config,
        "metrics":    {k: round(v, 6) if v is not None else None
                       for k, v in metrics.items()},
        "competence": {m: round(c, 6) for m, c in competence.items()},
        "per_sample": [
            {
                "sample_id": sample_ids[i],
                "gold":      gold[i],
                "pred":      labels[i],
                "entropy":   round(entropy[i], 6) if not math.isnan(entropy[i]) else None,
            }
            for i in range(len(sample_ids))
        ],
    }, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path.relative_to(ROOT)}")

    return {"metrics": metrics, "competence": competence, "entropy": entropy,
            "labels": labels, "sample_ids": sample_ids, "gold": gold}


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def run_sweep(lang: str, method: str, seed: int = 42) -> None:
    """Run hyperparameter sweep for one method and save mace_{method}_sweep_{lang}.csv + best params JSON."""
    sample_ids, gold, matrix, models = load_votes(lang)

    grid = SWEEP_GRID_VB if method == "vb" else SWEEP_GRID_EM

    # Build combos: VB adds ALPHA_BETA_PAIRS on top of base grid
    if method == "vb":
        base_combos = list(product(*grid.values()))
        combos = [
            {"n_restarts": r, "n_iter": i, "alpha": a, "beta": b}
            for (r, i), (a, b) in product(base_combos, ALPHA_BETA_PAIRS)
        ]
    else:
        keys   = list(grid.keys())
        combos = [dict(zip(keys, c)) for c in product(*grid.values())]

    print(f"\n{'='*60}")
    print(f"  MACE-{method.upper()} SWEEP  |  lang={lang.upper()}  ({len(combos)} configs)")
    print(f"{'='*60}")

    if method == "vb":
        header = f"  {'restarts':>8} {'n_iter':>6} {'alpha':>6} {'beta':>6}  {'BAcc':>7}  {'Kappa':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}"
    else:
        header = f"  {'restarts':>8} {'n_iter':>6} {'smooth':>7}  {'BAcc':>7}  {'Kappa':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}"
    print(header)
    print(f"  {'-'*85}")

    sweep_rows: list[dict] = []
    best_bacc, best_cfg, best_metrics = -1.0, {}, {}

    for cfg in combos:
        mace_kwargs = dict(method=method, n_restarts=cfg["n_restarts"],
                           n_iter=cfg["n_iter"], random_state=seed)
        if method == "vb":
            mace_kwargs["alpha"] = cfg["alpha"]
            mace_kwargs["beta"]  = cfg["beta"]
        else:
            mace_kwargs["smoothing"] = cfg["smoothing"]

        labels, _, _ = run_mace(models, matrix, **mace_kwargs)
        m    = compute_metrics(labels, gold)
        bacc = m["bacc"] or 0.0
        if bacc > best_bacc:
            best_bacc, best_cfg, best_metrics = bacc, cfg, m

        if method == "vb":
            print(
                f"  {cfg['n_restarts']:>8} {cfg['n_iter']:>6}"
                f" {cfg['alpha']:>6.1f} {cfg['beta']:>6.1f}"
                f"  {fmt(m['bacc'])}  {fmt(m['kappa'])}  {fmt(m['f1'])}  {fmt(m['precision'])}  {fmt(m['recall'])}"
            )
        else:
            print(
                f"  {cfg['n_restarts']:>8} {cfg['n_iter']:>6}"
                f" {cfg['smoothing']:>7.3f}"
                f"  {fmt(m['bacc'])}  {fmt(m['kappa'])}  {fmt(m['f1'])}  {fmt(m['precision'])}  {fmt(m['recall'])}"
            )
        sweep_rows.append({**cfg, **{k: round(v, 6) if v is not None else None
                                     for k, v in m.items()}})

    print(f"\n  Best BAcc: {best_bacc*100:.2f}%  →  {best_cfg}")

    out_dir = RESULTS_ROOT / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep CSV (tabular, easy to load with pandas)
    metric_cols  = ["bacc", "kappa", "f1", "precision", "recall", "tpr", "tnr", "n_valid"]
    csv_path     = out_dir / f"mace_{method}_sweep_{lang}.csv"
    vb_fields    = ["n_restarts", "n_iter", "alpha", "beta"]
    em_fields    = list(SWEEP_GRID_EM.keys())
    fieldnames   = (vb_fields if method == "vb" else em_fields) + metric_cols
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"  Saved sweep   → {csv_path.relative_to(ROOT)}")

    # Save sweep JSON with best_params key + all results
    json_path = out_dir / f"mace_{method}_sweep_{lang}.json"
    json_path.write_text(json.dumps({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "lang":        lang,
        "method":      method,
        "n_configs":   len(sweep_rows),
        "best_params": {
            "config":  {**best_cfg, "method": method, "seed": seed},
            "metrics": {k: round(v, 6) if v is not None else None
                        for k, v in best_metrics.items()},
        },
        "all_results": sweep_rows,
    }, indent=2), encoding="utf-8")
    print(f"  Saved JSON    → {json_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Control items sweep (semi-supervised)
# ---------------------------------------------------------------------------

CONTROL_FRACS = [0.1, 0.2, 0.3, 0.5]


def _plot_control_sweep(
    results: list[dict],
    lang: str,
    method: str,
    out_dir: Path,
    root: Path,
) -> None:
    import matplotlib.pyplot as plt

    fracs  = [r["control_frac"] for r in results]
    x      = [f * 100 for f in fracs]   # display as percentage
    labels = [f"{int(f*100)}%" for f in fracs]

    metrics = {
        "BAcc":      [r["bacc"]      for r in results],
        "Kappa":     [r["kappa"]     for r in results],
        "F1":        [r["f1"]        for r in results],
        "Precision": [r["precision"] for r in results],
        "Recall":    [r["recall"]    for r in results],
    }
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, vals), color in zip(metrics.items(), colors):
        y_vals = [v * 100 if v is not None else None for v in vals]
        ax.plot(x, y_vals, marker="o", label=name, color=color, linewidth=2, markersize=6)
        if name == "BAcc":
            for xi, yi in zip(x, y_vals):
                if yi is not None:
                    ax.annotate(f"{yi:.1f}%", xy=(xi, yi),
                                xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=8, color="black")

    # Baseline reference line at x=0
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Oracle control fraction (%)", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title(f"MACE-{method.upper()} semi-supervised control sweep — {lang.upper()}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"mace_{method}_control_sweep_{lang}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {out_path.relative_to(root)}")


def run_control_sweep(lang: str, method: str = "vb", seed: int = 42) -> None:
    """Vary control_frac and measure impact vs fully unsupervised baseline.

    Saves mace_{method}_control_sweep_{lang}.csv and .json.
    Uses default hyperparams (n_restarts=10, n_iter=50, alpha=0.5, beta=0.5).
    """
    _, gold, matrix, models = load_votes(lang)

    fracs   = [0.0] + CONTROL_FRACS  # 0.0 = unsupervised baseline
    results = []

    print(f"\n{'='*60}")
    print(f"  MACE-{method.upper()} CONTROL SWEEP  |  lang={lang.upper()}")
    print(f"  (oracle annotator injected for fraction of items)")
    print(f"{'='*60}")
    print(f"  {'control':>8}  {'n_ctrl':>7}  {'BAcc':>7}  {'Kappa':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
    print(f"  {'-'*70}")

    baseline_bacc = None
    for frac in fracs:
        run_matrix = matrix
        run_models = models
        if frac > 0.0:
            run_matrix = add_control_annotator(matrix, gold, frac=frac, seed=seed)
            run_models = models + [ORACLE_KEY]

        labels, competence, _ = run_mace(
            run_models, run_matrix,
            method=method, n_restarts=10, n_iter=50,
            alpha=0.5, beta=0.5, random_state=seed,
        )
        competence = {m: c for m, c in competence.items() if m != ORACLE_KEY}
        m_   = compute_metrics(labels, gold)
        bacc = m_["bacc"] or 0.0

        n_ctrl = sum(1 for g in gold if g is not None) * frac if frac > 0 else 0
        delta  = f"(+{(bacc - baseline_bacc)*100:.2f}pp)" if baseline_bacc is not None else "(baseline)"
        if baseline_bacc is None:
            baseline_bacc = bacc

        print(
            f"  {frac:>7.0%}  {int(n_ctrl):>7}  "
            f"{fmt(m_['bacc'])}  {fmt(m_['kappa'])}  {fmt(m_['f1'])}  "
            f"{fmt(m_['precision'])}  {fmt(m_['recall'])}  {delta}"
        )
        results.append({
            "control_frac": frac,
            "n_control_items": int(n_ctrl),
            **{k: round(v, 6) if v is not None else None for k, v in m_.items()},
            "competence": {m: round(c, 6) for m, c in competence.items()},
        })

    out_dir = RESULTS_ROOT / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV (flat — no competence)
    metric_cols = ["bacc", "kappa", "f1", "precision", "recall", "tpr", "tnr", "n_valid"]
    csv_path    = out_dir / f"mace_{method}_control_sweep_{lang}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["control_frac", "n_control_items"] + metric_cols,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved CSV  → {csv_path.relative_to(ROOT)}")

    # JSON (includes competence per run)
    json_path = out_dir / f"mace_{method}_control_sweep_{lang}.json"
    json_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lang": lang, "method": method,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"  Saved JSON → {json_path.relative_to(ROOT)}")

    _plot_control_sweep(results, lang, method, out_dir, ROOT)


# ---------------------------------------------------------------------------
# Viz — plot from existing saved CSVs
# ---------------------------------------------------------------------------

def _viz_control_sweep(lang: str) -> None:
    """Read saved control sweep CSVs and regenerate plots without rerunning MACE."""
    out_dir = RESULTS_ROOT / lang
    for method in ("vb", "em"):
        csv_path = out_dir / f"mace_{method}_control_sweep_{lang}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path.relative_to(ROOT)} not found — run --control-sweep first")
            continue
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            results = [
                {k: float(v) if v not in ("", None) else None for k, v in row.items()}
                for row in reader
            ]
        _plot_control_sweep(results, lang, method, out_dir, ROOT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone MACE analysis (VB + EM).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang", choices=LANGS)
    group.add_argument("--all", action="store_true", dest="all_langs")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sweep",         action="store_true", help="Sweep hyperparameters for both VB and EM")
    mode.add_argument("--control-sweep", action="store_true", help="Semi-supervised: vary oracle control fraction")
    mode.add_argument("--viz",           action="store_true", help="Plot existing control sweep CSVs without rerunning MACE")

    # Default single-run params
    parser.add_argument("--n-restarts",    default=10,    type=int)
    parser.add_argument("--n-iter",        default=50,    type=int)
    parser.add_argument("--smoothing",     default=0.005, type=float, help="EM-only. Paper default=0.01/num_labels=0.005 for binary")
    parser.add_argument("--default-noise", default=0.5,   type=float)
    parser.add_argument("--alpha",         default=0.5,   type=float, help="VB prior on spamming")
    parser.add_argument("--beta",          default=0.5,   type=float, help="VB prior on annotation dist")
    parser.add_argument("--threshold",     default=1.0,   type=float, help="Confidence threshold (1.0=keep all)")
    parser.add_argument("--control-frac",  default=0.0,   type=float, help="Fraction of items to use as oracle controls (0=unsupervised)")
    parser.add_argument("--seed",          default=42,    type=int)
    args = parser.parse_args()

    langs = LANGS if args.all_langs else [args.lang]

    for lang in langs:
        if args.sweep:
            run_sweep(lang, method="vb", seed=args.seed)
            run_sweep(lang, method="em", seed=args.seed)
        elif args.control_sweep:
            run_control_sweep(lang, method="vb", seed=args.seed)
            run_control_sweep(lang, method="em", seed=args.seed)
        elif args.viz:
            _viz_control_sweep(lang)
        else:
            # Load once, reuse for both methods
            sample_ids, gold, matrix, models = load_votes(lang)
            shared = dict(sample_ids=sample_ids, gold=gold, matrix=matrix, models=models)

            run_single(
                lang, method="vb",
                n_restarts=args.n_restarts, n_iter=args.n_iter,
                default_noise=args.default_noise,
                alpha=args.alpha, beta=args.beta,
                threshold=args.threshold, control_frac=args.control_frac,
                seed=args.seed, **shared,
            )
            run_single(
                lang, method="em",
                n_restarts=args.n_restarts, n_iter=args.n_iter,
                smoothing=args.smoothing, default_noise=args.default_noise,
                threshold=args.threshold, control_frac=args.control_frac,
                seed=args.seed, **shared,
            )


if __name__ == "__main__":
    main()
