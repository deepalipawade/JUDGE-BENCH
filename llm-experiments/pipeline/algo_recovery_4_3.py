"""
Algo recovery analysis on 4-3 majority failure cases.

For each language, loads the 4-3 failure cases and checks whether
DS / OWI / ISP produce the correct label on those specific samples.

Compares against:
  - majority vote       (already wrong by definition for these cases)
  - best_bacc LLM agg  (from the failure CSV)
  - random LLM agg     (from the failure CSV)
  - DS (Dawid-Skene)
  - OWI
  - ISP

Saves: results_tmp/memerag_ext/algo_recovery_4_3.md

Usage:
    python llm-experiments/pipeline/algo_recovery_4_3.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
LANGS        = ["en", "de", "es", "fr", "hi"]

LABEL_MAP = {"Supported": 1, "Not Supported": 0}


def load_failures(lang: str) -> list[dict]:
    """Load 4-3 failure rows from CSV for a language."""
    p = RESULTS_ROOT / lang / f"4_3_failures_{lang}.csv"
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sample_id", "").startswith("NOTE"):
                continue
            rows.append(row)
    return rows


def load_algo_labels(lang: str) -> dict[str, dict]:
    """Return {sample_id: {majority, owi, isp, ds, best_bacc, random}} from algo_agg file."""
    p = RESULTS_ROOT / lang / f"algo_agg_{lang}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    result = {}
    for rec in data.get("records", []):
        sid = rec.get("sample_id")
        if sid is None:
            continue
        result[sid] = {
            "majority": LABEL_MAP.get(rec.get("__majority")),
            "owi":      LABEL_MAP.get(rec.get("__owi")),
            "isp":      LABEL_MAP.get(rec.get("__isp")),
            "ds":       LABEL_MAP.get(rec.get("__ds")),
        }
    return result


def recovery(failures: list[dict], algo_labels: dict, method: str) -> tuple[int, int]:
    """Returns (n_correct, n_total) for a method over the failure cases."""
    n_correct = 0
    n_total = 0
    for row in failures:
        sid  = row["sample_id"]
        gold = int(row["gold_label"])

        if method in ("best_bacc", "random"):
            key = "best_bacc_correct" if method == "best_bacc" else "random_correct"
            val = row.get(key, "").strip().lower()
            if val not in ("true", "false"):
                continue
            n_total += 1
            if val == "true":
                n_correct += 1
        else:
            if sid not in algo_labels:
                continue
            pred = algo_labels[sid].get(method)
            if pred is None:
                continue
            n_total += 1
            if pred == gold:
                n_correct += 1

    return n_correct, n_total


def pct(n: int, d: int) -> str:
    if d == 0:
        return "—"
    return f"{n}/{d} ({100 * n / d:.1f}%)"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
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


METHODS = ["majority", "best_bacc", "random", "ds", "owi", "isp"]
METHOD_LABELS = {
    "majority": "Majority (wrong by def.)",
    "best_bacc": "Best-BAcc LLM agg",
    "random": "Random LLM agg",
    "ds": "Dawid-Skene",
    "owi": "OWI",
    "isp": "ISP",
}


def main() -> None:
    lines: list[str] = []
    lines.append("# Algo Recovery on 4-3 Majority Failure Cases\n")
    lines.append(
        "**Question:** When majority vote fails in a 4-3 split, do algo aggregation methods "
        "(DS, OWI, ISP) recover the correct answer more often?\n"
    )
    lines.append(
        "Recovery rate = (# cases where method predicted gold label) / (# 4-3 failure cases in that language).\n"
        "Majority is 0% by definition (these ARE the majority-wrong cases).\n"
    )

    all_counts: dict[str, dict[str, tuple[int, int]]] = {}

    for lang in LANGS:
        failures    = load_failures(lang)
        algo_labels = load_algo_labels(lang)

        if not failures:
            all_counts[lang] = {}
            continue

        counts: dict[str, tuple[int, int]] = {}
        for m in METHODS:
            counts[m] = recovery(failures, algo_labels, m)
        all_counts[lang] = counts

    # Per-language tables
    for lang in LANGS:
        counts = all_counts.get(lang, {})
        if not counts:
            lines.append(f"\n## {lang.upper()} — no failure cases found\n")
            continue

        n_failures = counts["majority"][1]
        lines.append(f"\n## {lang.upper()}  ({n_failures} failure cases)\n")

        rows = []
        for m in METHODS:
            nc, nt = counts[m]
            rows.append([METHOD_LABELS[m], pct(nc, nt)])
        lines += md_table(["Method", "Recovery (correct / total)"], rows)
        lines.append("")

    # Aggregate summary table (across all languages)
    lines.append("\n## Aggregate recovery across all languages\n")
    agg_rows = []
    for m in METHODS:
        total_correct = 0
        total_cases   = 0
        for lang in LANGS:
            counts = all_counts.get(lang, {})
            if m in counts:
                nc, nt = counts[m]
                total_correct += nc
                total_cases   += nt
        agg_rows.append([METHOD_LABELS[m], pct(total_correct, total_cases)])

    lines += md_table(["Method", "Recovery (correct / total, all langs)"], agg_rows)
    lines.append("")

    report = "\n".join(lines)
    print(report)
    out = RESULTS_ROOT / "algo_recovery_4_3.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
