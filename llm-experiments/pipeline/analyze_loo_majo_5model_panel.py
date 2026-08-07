"""
5-model panel analysis: fix 4 strong anchor judges, vary the 5th.

For each language, computes majority-vote BAcc for:
  - Full 7-model majority (baseline)
  - 4-anchor-only majority (2-2 ties excluded from BAcc)
  - Panel with anchors + GPT-OSS-120b as 5th
  - Panel with anchors + gemini-2.5-flash as 5th
  - Panel with anchors + Apertus-8B as 5th

Anchors (top 4 by avg BAcc across 5 languages):
  Qwen3.6-35B (80.8%), gemma-4-26b-a4b-it-maas (79.7%),
  MiniMax-M2.7 (77.7%), llama-3.3-70b-instruct-maas (76.4%)

Candidates for 5th seat:
  GPT-OSS-120b (76.0%), gemini-2.5-flash (75.8%), Apertus-8B (64.0%)

Saves: results_tmp/memerag_ext/panel5_analysis.md

Usage:
    python llm-experiments/pipeline/analyze_5model_panel.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
LANGS        = ["en", "de", "es", "fr", "hi"]

ANCHORS = [
    "Qwen3.6-35B",
    "google/gemma-4-26b-a4b-it-maas",
    "MiniMax-M2.7",
    "meta/llama-3.3-70b-instruct-maas",
]

CANDIDATES = [
    "GPT-OSS-120b",
    "gemini-2.5-flash",
    "Apertus-8B",
]

ALL_PROPOSERS = ANCHORS + CANDIDATES

# Best anchor model per language (by individual BAcc); used as 2-2 tiebreaker
LANG_BEST_MODEL = {
    "en": "Qwen3.6-35B",                    # 89.0%
    "de": "google/gemma-4-26b-a4b-it-maas", # 78.9%
    "es": "Qwen3.6-35B",                    # 78.8% (tied with gemma)
    "fr": "Qwen3.6-35B",                    # 79.7%
    "hi": "Qwen3.6-35B",                    # 80.2% (tied with gemma)
}

LABEL_MAP = {"Supported": 1, "Not Supported": 0}


def get_label(record: dict, model: str) -> int | None:
    out = record.get("model_outputs", {}).get(model)
    if out is None:
        return None
    return LABEL_MAP.get(out.get("label"))


def majority_label(labels: list[int | None]) -> int | None:
    valid = [l for l in labels if l is not None]
    if not valid:
        return None
    s = sum(valid)
    n = len(valid)
    if s > n / 2:
        return 1
    elif s < n / 2:
        return 0
    return None  # tie


def compute_bacc(pairs: list[tuple[int, int]]) -> tuple[float, int, int]:
    """Returns (bacc, n_valid, n_tied). pairs = (gold, pred) with no Nones."""
    tp = sum(1 for g, p in pairs if g == 1 and p == 1)
    tn = sum(1 for g, p in pairs if g == 0 and p == 0)
    fp = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn = sum(1 for g, p in pairs if g == 1 and p == 0)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (tpr + tnr) / 2, len(pairs), 0


def majority_class(records: list[dict]) -> int:
    """Returns the most frequent gold label (1 or 0) across all records."""
    gold_labels = [LABEL_MAP.get(r.get("gold_label")) for r in records]
    valid = [g for g in gold_labels if g is not None]
    return 1 if sum(valid) >= len(valid) / 2 else 0


def panel_bacc(
    records: list[dict],
    models: list[str],
    tie_break: str = "exclude",
    tie_break_model: str | None = None,
    maj_class: int | None = None,
) -> dict:
    """
    tie_break options:
      'exclude'       — tied samples are dropped from BAcc (original behaviour)
      'majority_class'— tied samples predict the most frequent gold label
      'best_model'    — tied samples use tie_break_model's prediction
    """
    pairs = []
    n_tied = 0
    n_missing = 0
    for rec in records:
        gold = LABEL_MAP.get(rec.get("gold_label"))
        if gold is None:
            n_missing += 1
            continue
        labels = [get_label(rec, m) for m in models]
        pred = majority_label(labels)
        if pred is None:
            n_tied += 1
            if tie_break == "majority_class":
                pred = maj_class
            elif tie_break == "best_model" and tie_break_model:
                pred = get_label(rec, tie_break_model)
            if pred is None:
                continue
        pairs.append((gold, pred))
    b, n_valid, _ = compute_bacc(pairs)
    return {"bacc": b, "n_valid": n_valid, "n_tied": n_tied, "n_missing": n_missing}


def fmt(b: float) -> str:
    return f"{b * 100:.1f}%"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build a padded, aligned markdown table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def pad(cell: str, w: int) -> str:
        return cell.ljust(w)
    header_row = "| " + " | ".join(pad(h, widths[i]) for i, h in enumerate(headers)) + " |"
    sep_row    = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
    data_rows  = [
        "| " + " | ".join(pad(cell, widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return [header_row, sep_row] + data_rows


def build_report() -> str:
    lines: list[str] = []
    lines.append("# 5-Model Panel Analysis\n")
    lines.append(
        "**Anchors (fixed):** Qwen3.6-35B, gemma-4-26b-a4b-it-maas, "
        "MiniMax-M2.7, llama-3.3-70b-instruct-maas\n"
    )
    lines.append(
        "**5th seat candidates:** GPT-OSS-120b (avg BAcc 76.0%), "
        "gemini-2.5-flash (75.8%), Apertus-8B (64.0%)\n"
    )
    lines.append(
        "BAcc = balanced accuracy of panel majority vote vs gold label. "
        "Tied votes (2-2 in 4-anchor panel) excluded from BAcc.\n"
    )

    summary: dict[str, dict] = {}
    main_rows: list[list[str]] = []

    for lang in LANGS:
        p = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"
        if not p.exists():
            main_rows.append([lang.upper(), "—", "—", "—", "—", "—", "—", "—", "—"])
            continue

        d = json.loads(p.read_text(encoding="utf-8"))
        records = d.get("records", [])
        n = len(records)

        best_model = LANG_BEST_MODEL[lang]
        r7   = panel_bacc(records, ALL_PROPOSERS)
        r4   = panel_bacc(records, ANCHORS, tie_break="exclude")
        r4bm = panel_bacc(records, ANCHORS, tie_break="best_model", tie_break_model=best_model)
        rGPT = panel_bacc(records, ANCHORS + ["GPT-OSS-120b"])
        rGEM = panel_bacc(records, ANCHORS + ["gemini-2.5-flash"])
        rAPT = panel_bacc(records, ANCHORS + ["Apertus-8B"])

        summary[lang] = {"7": r7, "4": r4, "4bm": r4bm, "GPT": rGPT, "GEM": rGEM, "APT": rAPT}

        main_rows.append([
            lang.upper(), str(n),
            fmt(r7["bacc"]),
            fmt(r4["bacc"]),
            fmt(r4bm["bacc"]),
            fmt(rGPT["bacc"]), fmt(rGEM["bacc"]), fmt(rAPT["bacc"]),
        ])

    lines.append(
        "4-anchor columns: (excl) = tied samples excluded; "
        "(best) = per-language best anchor breaks 2-2 ties "
        "(EN/ES/FR/HI=Qwen, DE=Gemma).\n"
    )
    lines += md_table(
        ["Lang", "n", "7-model", "4-anchor (excl)", "4-anchor (best)", "+GPT", "+Gemini", "+Apertus"],
        main_rows,
    )
    lines.append("")

    # Delta table
    lines.append("\n## Delta vs 4-anchor (per-language best tiebreaker) baseline (pp = percentage points)\n")
    delta_rows: list[list[str]] = []
    for lang in LANGS:
        if lang not in summary:
            delta_rows.append([lang.upper(), "—", "—", "—"])
            continue
        s  = summary[lang]
        b4 = s["4bm"]["bacc"]

        def delta(b: float) -> str:
            return f"{(b - b4) * 100:+.1f} pp"

        delta_rows.append([
            lang.upper(),
            delta(s["GPT"]["bacc"]),
            delta(s["GEM"]["bacc"]),
            delta(s["APT"]["bacc"]),
        ])

    lines += md_table(["Lang", "+GPT", "+Gemini", "+Apertus"], delta_rows)
    lines.append("")

    # Individual model BAcc reference table
    lines.append("\n## Individual model BAcc (reference)\n")
    model_bacc_ref = {
        "Qwen3.6-35B":               [89.0, 76.2, 78.8, 79.7, 80.2],
        "gemma-4-26b-a4b-it-maas":   [84.1, 78.9, 78.8, 76.3, 80.2],
        "MiniMax-M2.7":              [84.5, 71.9, 76.5, 76.9, 78.8],
        "llama-3.3-70b-instruct":    [81.3, 77.4, 76.7, 71.0, 75.8],
        "GPT-OSS-120b":              [82.1, 73.6, 73.9, 72.9, 77.9],
        "gemini-2.5-flash":          [78.0, 75.5, 75.3, 76.8, 73.5],
        "Apertus-8B":                [63.1, 59.6, 60.8, 61.5, 74.9],
    }
    ref_rows = []
    for m, vals in model_bacc_ref.items():
        avg = sum(vals) / len(vals)
        ref_rows.append([m] + [f"{v:.1f}%" for v in vals] + [f"{avg:.1f}%"])
    lines += md_table(["Model", "EN", "DE", "ES", "FR", "HI", "Avg"], ref_rows)

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    print(report)

    out = RESULTS_ROOT / "panel5_analysis.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
