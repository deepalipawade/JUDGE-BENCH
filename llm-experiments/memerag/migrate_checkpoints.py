"""
Migrates old memerag_judgement_{lang}_0.json checkpoints to the new per-sentence
sample_id format without re-running API calls where possible.

For each existing record (keyed by query_id in the old format):
  - Single-sentence query  → rename sample_id to query_id#s{sentence_id} (no API calls needed)
  - Multi-sentence query   → rescue the record by pointing it at the LAST sentence
                             (the one whose model_outputs are actually stored), update
                             gold_label / answer_segment / sentence_id to match.

After running this, execute judge_memerag.py --lang {lang} --all to fill in only
the missing non-last sentences of multi-sentence queries.

Usage:
    python memerag/migrate_checkpoints.py --langs de es fr hi
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT   = ROOT / "MEMERAG-main" / "data"
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"

sys.path.insert(0, str(Path(__file__).parent))
from judge_memerag import load_memerag_jsonl


def sentences_by_query(lang: str) -> dict[str, list[dict]]:
    """Returns {query_id: [sentence rows in processing order]} using the fixed loader."""
    data_file = DATA_ROOT / "memerag_ext" / f"{lang}.jsonl"
    rows = load_memerag_jsonl(data_file)
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    return by_query


def migrate_lang(lang: str) -> None:
    old_path = RESULTS_ROOT / lang / f"memerag_judgement_{lang}_0.json"
    new_path = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"

    if not old_path.exists():
        print(f"[{lang}] backup not found: {old_path} — skipping")
        return
    if new_path.exists():
        print(f"[{lang}] new checkpoint already exists — delete it first if you want to re-migrate")
        return

    by_query = sentences_by_query(lang)

    with old_path.open("r", encoding="utf-8") as f:
        old_data = json.load(f)
    old_records = [r for r in old_data.get("records", []) if isinstance(r, dict)]
    print(f"[{lang}] {len(old_records)} existing records in backup")

    migrated: dict[str, dict] = {}
    single_count = rescued_count = skipped_count = 0

    for rec in old_records:
        query_id = str(rec.get("query_id") or rec.get("sample_id", ""))
        sentences = by_query.get(query_id)
        if not sentences:
            print(f"  [WARN] query_id '{query_id}' not found in raw data — skipping")
            skipped_count += 1
            continue

        # The judge always processed sentences in list order; the last one wins in the dict.
        last = sentences[-1]
        new_id = last["sample_id"]

        new_rec = dict(rec)
        new_rec["sample_id"]                   = new_id
        new_rec["sentence_id"]                 = last["sentence_id"]
        new_rec["answer_segment"]              = last["answer_segment"]
        new_rec["gold_label"]                  = last["gold_label"]
        new_rec["gold_labels_all"]             = last["gold_labels_all"]
        new_rec["factuality_all"]              = last["factuality_all"]
        new_rec["relevance_all"]               = last["relevance_all"]
        new_rec["fine_grained_factuality_all"] = last["fine_grained_factuality_all"]
        new_rec["comments_all"]                = last["comments_all"]

        # Recompute correct_gold per model based on updated gold_label
        new_gold = last["gold_label"]
        for model_out in new_rec.get("model_outputs", {}).values():
            lbl = model_out.get("label")
            model_out["correct_gold"] = (lbl == new_gold) if lbl in {"Supported", "Not Supported"} else None

        migrated[new_id] = new_rec

        if len(sentences) == 1:
            single_count += 1
            print(f"  [exists] {query_id} → {new_id}")
        else:
            rescued_count += 1
            print(f"  [exists] {query_id} → {new_id}  (last of {len(sentences)} sentences; {len(sentences)-1} need fresh calls)")

    output = dict(old_data)
    output["records"]   = list(migrated.values())
    output["n_samples"] = len(migrated)

    with new_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_sentences  = sum(len(s) for s in by_query.values())
    fresh_needed     = total_sentences - len(migrated)
    models_per_rec   = 7

    print(f"[{lang}] single-sentence migrated : {single_count}")
    print(f"[{lang}] multi-sentence rescued   : {rescued_count}")
    if skipped_count:
        print(f"[{lang}] skipped (not in raw)    : {skipped_count}")
    print(f"[{lang}] saved {len(migrated)} records → {new_path.name}")
    print(f"[{lang}] fresh API calls still needed: {fresh_needed} records × {models_per_rec} models = {fresh_needed * models_per_rec}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate old checkpoints to per-sentence sample_id format.")
    parser.add_argument("--langs", nargs="+", default=["de", "es", "fr", "hi"],
                        help="Languages to migrate (default: de es fr hi)")
    args = parser.parse_args()
    for lang in args.langs:
        migrate_lang(lang)


if __name__ == "__main__":
    main()
