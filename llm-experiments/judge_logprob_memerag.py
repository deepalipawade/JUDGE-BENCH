"""
judge_logprob_memerag.py

Faithfulness judge using LOCAL HuggingFace models.
Runs the same SYSTEM_PROMPT + TASK_PROMPT as judge_memerag.py (rationale + <answer> tags),
then extracts log-probability confidence at the answer token position.

For each sample it saves:
  - label, rationale, raw output  (same as judge_memerag.py)
  - logprob_supported, logprob_not_supported   (raw log-softmax)
  - p_supported_normalized, p_not_supported_normalized  (re-normalised over the two labels)
  - label_mass  (how much probability mass fell on the two labels -- low = model deviated from format)
  - answer_token_step  (which generated-token step was the answer label)

Each model's output is saved to its own JSON file. Models are loaded and unloaded
one at a time to avoid GPU OOM on the cluster.

Usage:
    # run all models in MODELS list
    python judge_logprob_memerag.py --lang en

    # run a specific subset
    python judge_logprob_memerag.py --lang en --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct

    # single sample for debugging / supervisor demo
    python judge_logprob_memerag.py --lang en --sample-id 119#s0

    # restart a specific model from scratch
    python judge_logprob_memerag.py --lang en --models Qwen/Qwen2.5-7B-Instruct --no-resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT      = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "MEMERAG-main" / "data"

CHECKPOINT_EVERY = 5

# ── Prompt (identical to judge_memerag.py — no confidence tags) ──────────────

SYSTEM_PROMPT = (
    "Given a set of evidence passages, and an answer, determine if the answer is fully supported by the evidence passages or not. "
    "Analyze each sentence of the answer carefully and verify that all information it contains is explicitly stated in or can be directly inferred from the evidence passages.\n\n"
    'Output "Not Supported" if ANY of the following are true:\n\n'
    "The answer contains any information not explicitly stated in or directly inferable from the passages.\n"
    "The answer contradicts any information in the passages.\n"
    "The answer introduces any new information not found in the passages.\n"
    "The answer misrepresents or inaccurately paraphrases information from the passages.\n"
    "The answer draws conclusions not logically supported by the given information.\n"
    "The answer changes the level of certainty, specificity, or nuance from what is expressed in the passages.\n"
    "The answer does not directly address the specific aspect asked about in the question.\n"
    "The answer conflates or misrepresents separate pieces of information when summarizing multiple passages.\n\n"
    "Output Supported otherwise.\n\n"
    "Write your reasoning inside <rationale></rationale> tags.\n"
    "Provide your final answer inside <answer></answer> tags."
)

TASK_PROMPT = (
    "Evidence Passages:\n\n"
    "{context}\n\n"
    "Question:\n"
    "{query}\n\n"
    "Answer:\n"
    "{answer_segment}\n\n"
    "Provide your label, and rationale using the tags specified above."
)

LABELS = ["Supported", "Not Supported"]

# ── Model list ────────────────────────────────────────────────────────────────
# Edit this list before pushing to the cluster.
# Gated models (Llama, Gemma) require `huggingface-cli login` on the cluster node.
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "Qwen/Qwen2.5-32B-Instruct",
    "microsoft/Phi-4",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "tiiuae/Falcon3-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "01-ai/Yi-1.5-9B-Chat",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    # gated — need HF token:
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "google/gemma-2-9b-it",
]


# ── Data loading (same logic as judge_memerag.py) ────────────────────────────

def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s == "supported":
        return "Supported"
    if s == "not supported":
        return "Not Supported"
    return None


def normalize_annotation_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [] if value is None else [value]


def majority_vote_label(values: list[Any]) -> str | None:
    valid = [normalize_label(v) for v in values if normalize_label(v) is not None]
    if not valid:
        return None
    c = Counter(valid)
    if c["Supported"] == c["Not Supported"]:
        return None
    return c.most_common(1)[0][0]


def load_memerag_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            query_id = record.get("query_id")
            query    = str(record.get("query", "")).strip()
            context  = record.get("context", [])
            answers  = record.get("answer", [])

            if not query_id or not query or not isinstance(context, list) or not isinstance(answers, list):
                continue

            context_texts = [
                str(p.get("text", "")).strip()
                for p in context
                if isinstance(p, dict) and str(p.get("text", "")).strip()
            ]

            for s_idx, answer in enumerate(answers):
                if not isinstance(answer, dict):
                    continue
                sentence = str(answer.get("sentence", "")).strip()
                if not sentence:
                    continue

                factuality_all     = normalize_annotation_list(answer.get("factuality"))
                relevance_all      = normalize_annotation_list(answer.get("relevance"))
                fine_grained_all   = normalize_annotation_list(answer.get("fine_grained_factuality"))
                comments_all       = normalize_annotation_list(answer.get("comments"))
                gold_label         = normalize_label(answer.get("factuality"))
                if gold_label is None:
                    gold_label = majority_vote_label(factuality_all)

                sentence_id = answer.get("sentence_id", s_idx)
                rows.append({
                    "sample_id":               f"{query_id}#s{sentence_id}",
                    "query_id":                query_id,
                    "sentence_id":             sentence_id,
                    "query":                   query,
                    "context_texts":           context_texts,
                    "answer_segment":          sentence,
                    "gold_label":              gold_label,
                    "gold_labels_all":         factuality_all,
                    "factuality_all":          factuality_all,
                    "relevance_all":           relevance_all,
                    "fine_grained_factuality_all": fine_grained_all,
                    "comments_all":            comments_all,
                    "fine_grained_factuality": fine_grained_all[0] if fine_grained_all else answer.get("fine_grained_factuality"),
                    "relevance":               relevance_all[0] if relevance_all else answer.get("relevance"),
                    "comments":                comments_all[0] if comments_all else answer.get("comments"),
                })
    return rows


def build_prompt(row: dict[str, Any]) -> str:
    context = "\n".join(f"{i+1}. {t}" for i, t in enumerate(row["context_texts"]))
    return TASK_PROMPT.format(
        query=row["query"],
        context=context,
        answer_segment=row["answer_segment"],
    )


# ── Label extraction from generated text ─────────────────────────────────────

def extract_label_and_reason(text: str | None) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    m = re.search(r"<answer>\s*(supported|not supported)\s*</answer>", text, re.IGNORECASE)
    if m:
        label = normalize_label(m.group(1))
    else:
        lo = text.lower()
        label = "Not Supported" if "not supported" in lo else ("Supported" if "supported" in lo else None)
    if label is None:
        return None, None
    rm = re.search(r"<rationale>(.*?)</rationale>", text, re.IGNORECASE | re.DOTALL)
    rationale = rm.group(1).strip() if rm else None
    return label, rationale


# ── Token resolution ──────────────────────────────────────────────────────────

def resolve_label_tokens(tok: Any, labels: list[str]) -> dict[str, dict]:
    """Find the first token id for each label (handles ' Supported' vs 'Supported')."""
    resolved = {}
    for label in labels:
        best = None
        for variant in (label, " " + label):
            ids = tok.encode(variant, add_special_tokens=False)
            if ids and (best is None or len(ids) < len(best[1])):
                best = (variant, ids)
        resolved[label] = {"variant": best[0], "token_id": best[1][0]}
        print(f"  {label!r:18s} → first token {best[0]!r} (id {best[1][0]})")
    return resolved


# ── Log prob extraction from generation scores ───────────────────────────────

def find_answer_token_step(
    generated_ids: list[int],
    scores: tuple[torch.Tensor, ...],
    tok: Any,
    label_token_ids: dict[str, int],
) -> tuple[int | None, dict[str, float], dict[str, float], float]:
    """
    Scan generated tokens to find the first one that is a label token.
    Returns (step_index, raw_logprobs, normalized_probs, label_mass).
    step_index is None if no label token was found (format violation).
    """
    for step, token_id in enumerate(generated_ids):
        for lab, tid in label_token_ids.items():
            if token_id == tid:
                # Found the answer token — extract the distribution at this step
                logits    = scores[step][0]  # shape: (vocab,)
                log_probs = torch.log_softmax(logits.float(), dim=-1)

                raw_lp: dict[str, float] = {}
                for l2, t2 in label_token_ids.items():
                    raw_lp[l2] = log_probs[t2].item()

                # Re-normalise over the two label tokens
                label_logits = torch.stack([logits[t] for t in label_token_ids.values()]).float()
                normed       = torch.softmax(label_logits, dim=-1).tolist()
                norm_probs   = dict(zip(label_token_ids.keys(), normed))

                label_mass = sum(torch.tensor(lp).exp().item() for lp in raw_lp.values())
                return step, raw_lp, norm_probs, label_mass

    return None, {}, {}, 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def save_output(output_path: Path, header: dict, records: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({**header, "records": records}, fh, indent=2, ensure_ascii=False)


def run_one_model(
    model_name: str,
    candidate_data: list[dict],
    lang: str,
    data_file: Path,
    no_resume: bool,
    max_new_tokens: int,
    checkpoint_every: int,
) -> None:
    model_slug  = model_name.replace("/", "_").replace(".", "-")
    output_dir  = ROOT / "results_tmp" / "memerag_ext" / "logprob" / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"logprob_{model_slug}_{lang}.json"

    # ── Resume ───────────────────────────────────────────────────────────────
    completed: dict[str, dict] = {}
    if output_path.exists() and not no_resume:
        try:
            with output_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            for r in existing.get("records", []):
                if isinstance(r, dict) and r.get("sample_id"):
                    completed[str(r["sample_id"])] = r
            print(f"  [resume] Loaded {len(completed)} completed records.")
        except Exception as e:
            print(f"  [resume] Could not load existing output ({e}), starting fresh.")

    remaining = [r for r in candidate_data if str(r["sample_id"]) not in completed]
    print(f"  Remaining: {len(remaining)}")

    header = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "dataset_name": "memerag_ext",
        "lang":         lang,
        "model":        model_name,
        "data_file":    str(data_file),
        "n_samples":    len(candidate_data),
    }

    if not remaining:
        print("  Nothing to do — all samples already completed.")
        save_output(output_path, header, list(completed.values()))
        return

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"  Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"  Loaded on: {model.device}")

    print("  Resolving label tokens:")
    label_tokens    = resolve_label_tokens(tok, LABELS)
    label_token_ids = {lab: info["token_id"] for lab, info in label_tokens.items()}

    # ── Process samples ──────────────────────────────────────────────────────
    for idx, row in enumerate(remaining):
        sid = str(row["sample_id"])
        print(f"\n  [{idx+1}/{len(remaining)}] {sid}  gold={row['gold_label']}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(row)},
        ]
        prompt_str = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt_str, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len    = inputs["input_ids"].shape[1]
        generated_ids = out.sequences[0][prompt_len:].tolist()
        raw_text      = tok.decode(generated_ids, skip_special_tokens=True)

        label, rationale = extract_label_and_reason(raw_text)
        print(f"    label={label}  raw_len={len(raw_text)}")

        step, raw_lp, norm_probs, label_mass = find_answer_token_step(
            generated_ids, out.scores, tok, label_token_ids
        )

        if step is None:
            print("    [warn] answer token not found — label_mass=0")

        p_s_norm = norm_probs.get("Supported")
        if p_s_norm is not None:
            print(f"    p_supported_norm={p_s_norm:.4f}  label_mass={label_mass:.4f}")
        else:
            print("    [no log probs extracted]")

        completed[sid] = {
            **{k: row[k] for k in (
                "sample_id", "query_id", "sentence_id", "query", "answer_segment",
                "context_texts", "gold_label", "gold_labels_all", "factuality_all",
                "relevance_all", "fine_grained_factuality_all", "comments_all",
                "fine_grained_factuality", "relevance", "comments",
            )},
            "label":                      label,
            "rationale":                  rationale,
            "raw":                        raw_text,
            "logprob_supported":          raw_lp.get("Supported"),
            "logprob_not_supported":      raw_lp.get("Not Supported"),
            "p_supported_normalized":     norm_probs.get("Supported"),
            "p_not_supported_normalized": norm_probs.get("Not Supported"),
            "label_mass":                 label_mass,
            "answer_token_step":          step,
        }

        if (idx + 1) % checkpoint_every == 0:
            save_output(output_path, header, list(completed.values()))
            print(f"    [checkpoint] {len(completed)} records → {output_path.name}")

    save_output(output_path, header, list(completed.values()))
    print(f"\n  Saved {len(completed)} records → {output_path}")

    # Summary
    done    = [r for r in completed.values() if r.get("label") is not None]
    correct = sum(1 for r in done if r.get("label") == r.get("gold_label"))
    if done:
        print(f"  Accuracy: {correct}/{len(done)} = {correct/len(done)*100:.2f}%")
    low_mass = [r for r in done if (r.get("label_mass") or 1.0) < 0.3]
    print(f"  Low label_mass (<0.30): {len(low_mass)} samples")

    # ── Unload model to free GPU memory ──────────────────────────────────────
    del model, tok
    torch.cuda.empty_cache()
    print(f"  Unloaded {model_name}.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log-prob faithfulness judge (local HF models)")
    parser.add_argument("--lang",      default="en",  help="Language code (default: en)")
    parser.add_argument("--models",    nargs="+", default=None,
                        help="Model(s) to run. Defaults to full MODELS list in script.")
    parser.add_argument("--sample-id", default=None,  help="Run a single sample by ID (e.g. 119#s0)")
    parser.add_argument("--samples",   type=int, default=None, help="Random subset size")
    parser.add_argument("--no-resume", action="store_true",    help="Start fresh, ignore existing output")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_file = DATA_ROOT / "memerag_ext" / f"{args.lang}.jsonl"
    if not data_file.exists():
        sys.exit(f"Dataset not found: {data_file}")

    all_data = load_memerag_jsonl(data_file)
    if not all_data:
        sys.exit("No valid examples loaded.")

    if args.sample_id:
        candidate_data = [r for r in all_data if str(r["sample_id"]) == args.sample_id]
        if not candidate_data:
            sys.exit(f"Sample ID not found: {args.sample_id}")
    elif args.samples is not None:
        import random; random.seed(42)
        random.shuffle(all_data)
        candidate_data = all_data[: args.samples]
    else:
        candidate_data = all_data

    print(f"Dataset: {len(candidate_data)} samples  ({data_file.name})")

    model_list = args.models if args.models else MODELS
    print(f"Models to run ({len(model_list)}): {model_list}\n")

    for model_name in model_list:
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        run_one_model(
            model_name      = model_name,
            candidate_data  = candidate_data,
            lang            = args.lang,
            data_file       = data_file,
            no_resume       = args.no_resume,
            max_new_tokens  = args.max_new_tokens,
            checkpoint_every= args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
