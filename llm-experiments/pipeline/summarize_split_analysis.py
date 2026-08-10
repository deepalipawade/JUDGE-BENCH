"""
Summary tables for vote-confidence analysis across all languages and split levels.

Reads: results_tmp/memerag_ext/{lang}/analysis_vote_confidence_{lang}.json
Saves: results_tmp/memerag_ext/split_analysis_summary.md

Usage:
    python pipeline/summarize_split_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
LANGS        = ["en", "de", "es", "fr", "hi"]
SPLITS       = ["6-1", "5-2", "4-3"]


def pct(n: int, total: int) -> str:
    return f"{round(n / total * 100, 1)}%" if total else "—"


def build_tables() -> str:
    lines: list[str] = []

    for split in SPLITS:
        lines.append(f"### Split {split} — cases where majority was wrong\n")
        header = (
            "| Lang | n_failures | better_judges_outvoted | "
            "best_bacc_recovered | random_recovered | neither_recovered |"
        )
        sep = "|------|------------|------------------------|---------------------|------------------|-------------------|"
        lines.append(header)
        lines.append(sep)

        for lang in LANGS:
            p = RESULTS_ROOT / lang / f"analysis_vote_confidence_{lang}.json"
            if not p.exists():
                lines.append(f"| {lang.upper()} | — | — | — | — | — |")
                continue

            d  = json.loads(p.read_text(encoding="utf-8"))
            v  = d.get("q2_majority_wrong_by_split", {}).get(split, {})
            n  = v.get("n_cases", 0)

            if not v or n == 0:
                lines.append(f"| {lang.upper()} | 0 | — | — | — | — |")
                continue

            n_bj      = v["n_minority_stronger"]
            n_best    = v["best_bacc_correct"]
            n_rand    = v["random_correct"]
            cases     = v.get("cases", [])
            n_neither = sum(
                1 for c in cases
                if not c.get("best_bacc_correct") and not c.get("random_correct")
            )

            lines.append(
                f"| {lang.upper()} | {n} "
                f"| {n_bj}/{n} ({pct(n_bj, n)}) "
                f"| {n_best}/{n} ({pct(n_best, n)}) "
                f"| {n_rand}/{n} ({pct(n_rand, n)}) "
                f"| {n_neither}/{n} ({pct(n_neither, n)}) |"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    content = build_tables()

    print(content)

    out_path = RESULTS_ROOT / "split_analysis_summary.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
