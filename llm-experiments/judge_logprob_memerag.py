"""
judge_logprob_memerag.py

Faithfulness judge using LOCAL HuggingFace models.
Extracts log-probability confidence at the label decision point.

Two modes (controlled by --rationale flag):

  DEFAULT (bare-label):
    Prompt asks for "Supported" or "Not Supported" directly, no explanation.
    Logprob is read at step 0 — the model's uncertainty BEFORE any reasoning.
    Output file: logprob_judgement_{lang}.json

  --rationale mode:
    Prompt asks for <rationale>...</rationale> then <answer>...</answer>.
    Logprob is read at the step where the label token appears inside <answer>.
    Warning: after writing a rationale, distributions collapse to ~0/1 for most
    models, making the confidence score less informative than bare-label mode.
    Output file: logprob_judgement_{lang}_rationale.json

Usage:
    # bare-label mode (recommended for calibrated logprobs)
    python judge_logprob_memerag.py --lang en

    # rationale mode (preserves reasoning, but logprobs less informative)
    python judge_logprob_memerag.py --lang en --rationale

    # specific models only
    python judge_logprob_memerag.py --lang en --models meta-llama/Llama-3.1-8B-Instruct

    # single sample for debugging
    python judge_logprob_memerag.py --lang en --sample-id 119#s0

    # 2-sample test without touching existing data
    python judge_logprob_memerag.py --lang en --models meta-llama/Llama-3.1-8B-Instruct --samples 2

    # restart a specific model from scratch
    python judge_logprob_memerag.py --lang en --models Qwen/Qwen3-8B --no-resume
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

import time

ROOT      = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "MEMERAG-main" / "data"

CHECKPOINT_EVERY = 5
MAX_RETRIES      = 3
RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry

# ── Prompts ───────────────────────────────────────────────────────────────────

# Default: bare-label (no rationale)
# Logprob at step 0 = model's uncertainty before any reasoning.
SYSTEM_PROMPT_BARE = (
    "Given a set of evidence passages, and an answer, determine if the answer "
    "is fully supported by the evidence passages or not. "
    "Analyze each sentence of the answer carefully and verify that all "
    "information it contains is explicitly stated in or can be directly "
    "inferred from the evidence passages.\n\n"
    "Output Not Supported if ANY of the following are true:\n\n"
    "The answer contains any information not explicitly stated in or directly inferable from the passages.\n"
    "The answer contradicts any information in the passages.\n"
    "The answer introduces any new information not found in the passages.\n"
    "The answer misrepresents or inaccurately paraphrases information from the passages.\n"
    "The answer draws conclusions not logically supported by the given information.\n"
    "The answer changes the level of certainty, specificity, or nuance from what is expressed in the passages.\n"
    "The answer does not directly address the specific aspect asked about in the question.\n"
    "The answer conflates or misrepresents separate pieces of information when summarizing multiple passages.\n\n"
    "Output Supported otherwise.\n\n"
    "Reply with exactly one word or phrase: Supported or Not Supported. No explanation."
)

TASK_PROMPT_BARE = (
    "Evidence Passages:\n\n"
    "{context}\n\n"
    "Question:\n"
    "{query}\n\n"
    "Answer:\n"
    "{answer_segment}\n\n"
    "Your label (Supported or Not Supported):"
)

# --rationale mode: rationale then <answer> tag
# Logprob at the step where the label appears inside <answer>.
# Note: distributions typically collapse to ~0/1 after reasoning is written.
SYSTEM_PROMPT_RATIONALE = (
    "Given a set of evidence passages, and an answer, determine if the answer is fully "
    "supported by the evidence passages or not. "
    "Analyze each sentence of the answer carefully and verify that all information it "
    "contains is explicitly stated in or can be directly inferred from the evidence passages.\n\n"
    "Output Not Supported if ANY of the following are true:\n\n"
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
    "Provide your final answer inside <answer></answer> tags as exactly "
    "<answer>Supported</answer> or <answer>Not Supported</answer>.\n\n"
    "Express your confidence based on EVIDENCE QUALITY in the passages, not on how certain you feel about your label.\n"
    "If the answer has multiple sentences with different evidence quality, base your confidence on "
    "whichever sentence most drove your final label — typically the weakest-supported sentence if the "
    "label is Not Supported, or the least-direct sentence if the label is Supported. Do not average across sentences.\n"
    "Output exactly one of these three fixed scores — do not interpolate:\n\n"
    "  High   → <confidence>High</confidence>   <confidence_score>0.95</confidence_score>\n"
    "           At least one passage directly and explicitly addresses the decisive sentence/claim\n"
    "           (confirms or refutes it with clear textual evidence).\n"
    "           e.g. Confirm: the claim's key fact is stated word-for-word in a passage.\n"
    "           e.g. Refute:  a passage directly states the opposite of what the claim says.\n\n"
    "  Medium → <confidence>Medium</confidence> <confidence_score>0.65</confidence_score>\n"
    "           Passages are topically relevant but the link requires inference or paraphrasing.\n"
    "           e.g. Confirm: a passage implies the claim is true but does not state it outright.\n"
    "           e.g. Refute:  a passage implies the claim is false but does not explicitly say so.\n\n"
    "  Low    → <confidence>Low</confidence>    <confidence_score>0.35</confidence_score>\n"
    "           No passage addresses the specific claim — passages cover the topic area\n"
    "           but are silent on the specific fact; your label is a best guess.\n\n"
    "Output format (always all four tags, in this order):\n"
    "<rationale>...</rationale>\n"
    "<answer>Supported OR Not Supported</answer>\n"
    "<confidence>High OR Medium OR Low</confidence>\n"
    "<confidence_score>0.95 OR 0.65 OR 0.35</confidence_score>\n"
)

TASK_PROMPT_RATIONALE = (
    "Evidence Passages:\n\n"
    "{context}\n\n"
    "Question:\n"
    "{query}\n\n"
    "Answer:\n"
    "{answer_segment}\n\n"
    "Provide your rationale, label, and confidence using the tags specified above."
)

# --chain mode: initial answer → rationale → final answer → confidence
# Pre-reasoning logprob at <answer_initial>; post-reasoning logprob at <answer>.
# Flip detection: answer_initial != answer.
SYSTEM_PROMPT_CHAIN = (
    "Given a set of evidence passages, and an answer, determine if the answer is fully "
    "supported by the evidence passages or not. "
    "Analyze each sentence of the answer carefully and verify that all information it "
    "contains is explicitly stated in or can be directly inferred from the evidence passages.\n\n"
    "Output Not Supported if ANY of the following are true:\n\n"
    "The answer contains any information not explicitly stated in or directly inferable from the passages.\n"
    "The answer contradicts any information in the passages.\n"
    "The answer introduces any new information not found in the passages.\n"
    "The answer misrepresents or inaccurately paraphrases information from the passages.\n"
    "The answer draws conclusions not logically supported by the given information.\n"
    "The answer changes the level of certainty, specificity, or nuance from what is expressed in the passages.\n"
    "The answer does not directly address the specific aspect asked about in the question.\n"
    "The answer conflates or misrepresents separate pieces of information when summarizing multiple passages.\n\n"
    "Output Supported otherwise.\n\n"
    "Express your confidence based on EVIDENCE QUALITY in the passages, not on how certain you feel about your label.\n"
    "If the answer has multiple sentences with different evidence quality, base your confidence on "
    "whichever sentence most drove your final label — typically the weakest-supported sentence if the "
    "label is Not Supported, or the least-direct sentence if the label is Supported. Do not average across sentences.\n"
    "Output exactly one of these three fixed scores — do not interpolate:\n\n"
    "  High   → <confidence>High</confidence>   <confidence_score>0.95</confidence_score>\n"
    "           At least one passage directly and explicitly addresses the decisive sentence/claim\n"
    "           (confirms or refutes it with clear textual evidence).\n\n"
    "  Medium → <confidence>Medium</confidence> <confidence_score>0.65</confidence_score>\n"
    "           Passages are topically relevant but the link requires inference or paraphrasing.\n\n"
    "  Low    → <confidence>Low</confidence>    <confidence_score>0.35</confidence_score>\n"
    "           No passage addresses the specific claim — your label is a best guess.\n\n"
    "Output format (always all five tags, in this order):\n"
    "1. Your initial label BEFORE reasoning (gut decision):\n"
    "   <answer_initial>Supported OR Not Supported</answer_initial>\n\n"
    "2. Your reasoning through the evidence:\n"
    "   <rationale>...</rationale>\n\n"
    "3. Your final label — you MAY revise your initial answer if reasoning changes your mind:\n"
    "   <answer>Supported OR Not Supported</answer>\n\n"
    "4. Your confidence based on EVIDENCE QUALITY:\n"
    "   <confidence>High OR Medium OR Low</confidence>\n"
    "   <confidence_score>0.95 OR 0.65 OR 0.35</confidence_score>\n"
)

TASK_PROMPT_CHAIN = (
    "Evidence Passages:\n\n"
    "{context}\n\n"
    "Question:\n"
    "{query}\n\n"
    "Answer:\n"
    "{answer_segment}\n\n"
    "Provide your initial label, rationale, final label, and confidence using the tags specified above."
)

LABELS = ["Supported", "Not Supported"]

# ── Model list ────────────────────────────────────────────────────────────────
# Edit this list before pushing to the cluster.
# Gated models (Llama, Gemma) require `huggingface-cli login` on the cluster node.
MODELS = [
    # "Qwen/Qwen2.5-7B-Instruct", # Qwen 3 or more
    # "Qwen/Qwen2.5-14B-Instruct",
    # "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen3-8B", #qwen thinking disabled
    "microsoft/Phi-4",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # gated — need HF token:
    "meta-llama/Llama-3.1-8B-Instruct", #gated
    "google/gemma-2-9b-it", #gated

    # "microsoft/Phi-4-mini-instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # "CohereForAI/c4ai-command-r-v01",
    # "allenai/OLMo-7B-Instruct",
    # "tiiuae/Falcon3-7B-Instruct",
    # "allenai/OLMo-2-1124-7B-Instruct",
    # "01-ai/Yi-1.5-9B-Chat",
    # "NousResearch/Hermes-3-Llama-3.1-8B",
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


# ── Label extraction from generated text ─────────────────────────────────────

def extract_label_bare(text: str | None) -> tuple[str | None, str | None]:
    """Extract label from bare-label output (no rationale expected)."""
    if not text:
        return None, None
    lo = text.strip().lower()
    if "not supported" in lo:
        return "Not Supported", None
    if "supported" in lo:
        return "Supported", None
    return None, None


def extract_label_chain(text: str | None) -> tuple[str | None, str | None, str | None, str | None, float | None]:
    """Extract initial label, final label, rationale, confidence, confidence_score from chain output."""
    if not text:
        return None, None, None, None, None
    m_init = re.search(r"<answer_initial>\s*(supported|not supported)\s*</answer_initial>", text, re.IGNORECASE)
    label_initial = normalize_label(m_init.group(1)) if m_init else None
    m_final = re.search(r"<answer>\s*(supported|not supported)\s*</answer>", text, re.IGNORECASE)
    if m_final:
        label_final = normalize_label(m_final.group(1))
    else:
        lo = text.lower()
        label_final = "Not Supported" if "not supported" in lo else ("Supported" if "supported" in lo else None)
    rm = re.search(r"<rationale>(.*?)</rationale>", text, re.IGNORECASE | re.DOTALL)
    rationale = rm.group(1).strip() if rm else None
    cm = re.search(r"<confidence>\s*(high|medium|low)\s*</confidence>", text, re.IGNORECASE)
    confidence = cm.group(1).capitalize() if cm else None
    sm = re.search(r"<confidence_score>\s*([0-9.]+)\s*</confidence_score>", text, re.IGNORECASE)
    confidence_score = float(sm.group(1)) if sm else None
    return label_initial, label_final, rationale, confidence, confidence_score


def extract_label_rationale(text: str | None) -> tuple[str | None, str | None, str | None, float | None]:
    """Extract label, rationale, verbalized confidence, and confidence score from tagged output."""
    if not text:
        return None, None, None, None
    m = re.search(r"<answer>\s*(supported|not supported)\s*</answer>", text, re.IGNORECASE)
    if m:
        label = normalize_label(m.group(1))
    else:
        lo = text.lower()
        label = "Not Supported" if "not supported" in lo else ("Supported" if "supported" in lo else None)
    if label is None:
        return None, None, None, None
    rm = re.search(r"<rationale>(.*?)</rationale>", text, re.IGNORECASE | re.DOTALL)
    rationale = rm.group(1).strip() if rm else None
    cm = re.search(r"<confidence>\s*(high|medium|low)\s*</confidence>", text, re.IGNORECASE)
    confidence = cm.group(1).capitalize() if cm else None
    sm = re.search(r"<confidence_score>\s*([0-9.]+)\s*</confidence_score>", text, re.IGNORECASE)
    confidence_score = float(sm.group(1)) if sm else None
    return label, rationale, confidence, confidence_score


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

def extract_step0_logprobs(
    scores: tuple[torch.Tensor, ...],
    label_token_ids: dict[str, int],
) -> tuple[int, dict[str, float], dict[str, float], float]:
    """
    Read the label distribution at step 0 (the very first generated token).
    Used in bare-label mode: the first token IS the label, so this captures
    the model's uncertainty before any reasoning has narrowed the distribution.
    """
    logits    = scores[0][0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    raw_lp    = {l: log_probs[tid].item() for l, tid in label_token_ids.items()}
    label_logits = torch.stack([logits[t] for t in label_token_ids.values()]).float()
    normed       = torch.softmax(label_logits, dim=-1).tolist()
    norm_probs   = dict(zip(label_token_ids.keys(), normed))
    label_mass   = sum(torch.tensor(lp).exp().item() for lp in raw_lp.values())
    return 0, raw_lp, norm_probs, label_mass


def find_answer_token_step(
    generated_ids: list[int],
    scores: tuple[torch.Tensor, ...],
    tok: Any,
    label_token_ids: dict[str, int],
    tag: str = "<answer>",
) -> tuple[int | None, dict[str, float], dict[str, float], float]:
    """
    Find the step where the label word begins inside the given tag.
    Used in --rationale and --chain modes. Decodes tokens incrementally
    and triggers at the step where text after `tag` starts with 'sup' or 'not'.
    Falls back to exact token-ID scan if the tag is not found.
    `tag` should be '<answer>' (default) or '<answer_initial>' for chain mode.
    """
    tag_lo  = tag.lower()
    tag_len = len(tag_lo)
    cumulative = ""
    for step, token_id in enumerate(generated_ids):
        piece = tok.decode([token_id], skip_special_tokens=False)
        cumulative += piece
        lo = cumulative.lower()
        if tag_lo not in lo:
            continue
        after = lo[lo.rfind(tag_lo) + tag_len:].lstrip()
        if after.startswith("sup") or after.startswith("not"):
            logits    = scores[step][0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            raw_lp    = {l: log_probs[label_token_ids[l]].item() for l in label_token_ids}
            label_logits = torch.stack([logits[t] for t in label_token_ids.values()]).float()
            normed       = torch.softmax(label_logits, dim=-1).tolist()
            norm_probs   = dict(zip(label_token_ids.keys(), normed))
            label_mass   = sum(torch.tensor(lp).exp().item() for lp in raw_lp.values())
            return step, raw_lp, norm_probs, label_mass

    # Fallback: exact token-ID scan
    for step, token_id in enumerate(generated_ids):
        for lab, tid in label_token_ids.items():
            if token_id == tid:
                logits    = scores[step][0]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                raw_lp    = {l2: log_probs[t2].item() for l2, t2 in label_token_ids.items()}
                label_logits = torch.stack([logits[t] for t in label_token_ids.values()]).float()
                normed       = torch.softmax(label_logits, dim=-1).tolist()
                norm_probs   = dict(zip(label_token_ids.keys(), normed))
                label_mass   = sum(torch.tensor(lp).exp().item() for lp in raw_lp.values())
                return step, raw_lp, norm_probs, label_mass

    return None, {}, {}, 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def save_output(output_path: Path, header: dict, records: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({**header, "records": list(records)}, fh, indent=2, ensure_ascii=False)


def load_existing(output_path: Path) -> dict[str, dict]:
    """Load existing JSON and return {sample_id: record} dict."""
    if not output_path.exists():
        return {}
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
        return {
            str(r["sample_id"]): r
            for r in existing.get("records", [])
            if isinstance(r, dict) and r.get("sample_id")
        }
    except Exception as e:
        print(f"  [resume] Could not load existing output ({e}), starting fresh.")
        return {}


def apply_chat_template_safe(tok: Any, messages: list[dict]) -> str:
    """
    Apply chat template with two fallbacks:
    1. TypeError  → enable_thinking not supported (non-Qwen models); retry without it.
    2. TemplateError / 'system' in error → system role not supported (Gemma etc.);
       merge system prompt into the user turn and retry.
    """
    for use_thinking in (True, False):
        kwargs = {"enable_thinking": False} if use_thinking else {}
        for merge_system in (False, True):
            msgs = messages
            if merge_system:
                merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
                msgs = [{"role": "user", "content": merged}]
            try:
                return tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except TypeError:
                break  # enable_thinking not supported → try without it
            except Exception as e:
                if "system" in str(e).lower():
                    continue  # system role not supported → try merged
                raise
    raise RuntimeError("apply_chat_template failed with all fallback attempts")


def run_one_model(
    model_name: str,
    candidate_data: list[dict],
    all_records: dict[str, dict],   # shared {sample_id: record} updated in-place
    output_path: Path,
    header: dict,
    no_resume: bool,
    max_new_tokens: int,
    checkpoint_every: int,
    use_rationale: bool = False,
    use_chain: bool = False,
) -> None:
    # Find samples where this model's output is missing
    remaining = []
    for row in candidate_data:
        sid = str(row["sample_id"])
        existing_rec = all_records.get(sid, {})
        if no_resume or model_name not in existing_rec.get("model_outputs", {}):
            remaining.append(row)

    print(f"  Remaining for {model_name}: {len(remaining)}")
    if not remaining:
        print("  Nothing to do — skipping model load.\n")
        return

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"  Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"  Loaded on: {model.device}")

    if use_chain:
        system_prompt = SYSTEM_PROMPT_CHAIN
        task_prompt   = TASK_PROMPT_CHAIN
        mode_label    = "chain"
    elif use_rationale:
        system_prompt = SYSTEM_PROMPT_RATIONALE
        task_prompt   = TASK_PROMPT_RATIONALE
        mode_label    = "rationale"
    else:
        system_prompt = SYSTEM_PROMPT_BARE
        task_prompt   = TASK_PROMPT_BARE
        mode_label    = "bare-label"
    print(f"  Mode: {mode_label}")

    print("  Resolving label tokens:")
    label_tokens    = resolve_label_tokens(tok, LABELS)
    label_token_ids = {lab: info["token_id"] for lab, info in label_tokens.items()}

    # ── Process samples ──────────────────────────────────────────────────────
    for idx, row in enumerate(remaining):
        sid = str(row["sample_id"])
        print(f"\n  [{idx+1}/{len(remaining)}] {sid}  gold={row['gold_label']}")

        label = rationale = raw_text = None
        label_initial = None
        confidence = confidence_score = None
        step, raw_lp, norm_probs, label_mass = None, {}, {}, 0.0
        step_initial, raw_lp_initial, norm_probs_initial, label_mass_initial = None, {}, {}, 0.0
        error_msg = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                context = "\n".join(f"{i+1}. {t}" for i, t in enumerate(row["context_texts"]))
                user_content = task_prompt.format(
                    query=row["query"],
                    context=context,
                    answer_segment=row["answer_segment"],
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ]
                prompt_str = apply_chat_template_safe(tok, messages)
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

                if use_chain:
                    label_initial, label, rationale, confidence, confidence_score = extract_label_chain(raw_text)
                    step_initial, raw_lp_initial, norm_probs_initial, label_mass_initial = find_answer_token_step(
                        generated_ids, out.scores, tok, label_token_ids, tag="<answer_initial>"
                    )
                    step, raw_lp, norm_probs, label_mass = find_answer_token_step(
                        generated_ids, out.scores, tok, label_token_ids, tag="<answer>"
                    )
                    if step_initial is None:
                        print("    [warn] <answer_initial> token not found")
                    if step is None:
                        print("    [warn] <answer> (final) token not found")
                elif use_rationale:
                    label, rationale, confidence, confidence_score = extract_label_rationale(raw_text)
                    step, raw_lp, norm_probs, label_mass = find_answer_token_step(
                        generated_ids, out.scores, tok, label_token_ids
                    )
                    if step is None:
                        print("    [warn] answer token not found inside <answer> tag")
                else:
                    label, rationale = extract_label_bare(raw_text)
                    step, raw_lp, norm_probs, label_mass = extract_step0_logprobs(
                        out.scores, label_token_ids
                    )

                error_msg = None
                break  # success

            except Exception as e:
                error_msg = str(e)
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"    [retry {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e} — waiting {delay}s")
                    torch.cuda.empty_cache()
                    time.sleep(delay)
                else:
                    print(f"    [failed after {MAX_RETRIES} attempts] {type(e).__name__}: {e}")

        if error_msg:
            print(f"    Skipping sample — saved as error record.")
        else:
            p_s_norm = norm_probs.get("Supported")
            if use_chain and label_initial is not None:
                flip = label_initial != label if (label_initial and label) else None
                p_init = norm_probs_initial.get("Supported")
                print(f"    label_initial={label_initial}  label={label}  flipped={flip}  raw_len={len(raw_text) if raw_text else 0}")
                if p_init is not None:
                    print(f"    p_supported_initial={p_init:.4f}  label_mass_initial={label_mass_initial:.4f}")
            else:
                print(f"    label={label}  raw_len={len(raw_text) if raw_text else 0}")
            if p_s_norm is not None:
                print(f"    p_supported_norm={p_s_norm:.4f}  label_mass={label_mass:.4f}")
                if label_mass < 0.3:
                    print("    [warn] low label_mass — model may have output something unexpected")

        # Merge into shared record (create if first model for this sample)
        if sid not in all_records:
            all_records[sid] = {
                k: row[k] for k in (
                    "sample_id", "query_id", "sentence_id", "query", "answer_segment",
                    "context_texts", "gold_label", "gold_labels_all", "factuality_all",
                    "relevance_all", "fine_grained_factuality_all", "comments_all",
                    "fine_grained_factuality", "relevance", "comments",
                )
            }
            all_records[sid]["model_outputs"] = {}

        out_rec = {
            "label":                      label,
            "rationale":                  rationale,
            "verbalized_confidence":      confidence,
            "verbalized_confidence_score": confidence_score,
            "raw":                        raw_text,
            "logprob_supported":          raw_lp.get("Supported"),
            "logprob_not_supported":      raw_lp.get("Not Supported"),
            "p_supported_normalized":     norm_probs.get("Supported"),
            "p_not_supported_normalized": norm_probs.get("Not Supported"),
            "label_mass":                 label_mass,
            "answer_token_step":          step,
            "error":                      error_msg,
        }
        if use_chain:
            out_rec.update({
                "label_initial":                      label_initial,
                "flipped":                            (label_initial != label) if (label_initial and label) else None,
                "logprob_supported_initial":          raw_lp_initial.get("Supported"),
                "logprob_not_supported_initial":      raw_lp_initial.get("Not Supported"),
                "p_supported_normalized_initial":     norm_probs_initial.get("Supported"),
                "p_not_supported_normalized_initial": norm_probs_initial.get("Not Supported"),
                "label_mass_initial":                 label_mass_initial or None,
                "answer_initial_token_step":          step_initial,
            })
        all_records[sid]["model_outputs"][model_name] = out_rec

        if (idx + 1) % checkpoint_every == 0:
            save_output(output_path, header, list(all_records.values()))
            print(f"    [checkpoint] {len(all_records)} records → {output_path.name}")

    save_output(output_path, header, list(all_records.values()))

    # Per-model summary
    done    = [r for r in all_records.values() if model_name in r.get("model_outputs", {})]
    correct = sum(
        1 for r in done
        if r["model_outputs"][model_name].get("label") == r.get("gold_label")
    )
    if done:
        print(f"\n  {model_name} accuracy: {correct}/{len(done)} = {correct/len(done)*100:.2f}%")
    low_mass = [
        r for r in done
        if (r["model_outputs"][model_name].get("label_mass") or 1.0) < 0.3
    ]
    print(f"  Low label_mass (<0.30): {len(low_mass)} samples")

    # ── Unload model ─────────────────────────────────────────────────────────
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
    parser.add_argument("--rationale", action="store_true",
                        help="rationale → answer → confidence. Logprob at <answer> step (post-reasoning).")
    parser.add_argument("--chain", action="store_true",
                        help="answer_initial → rationale → answer → confidence. "
                             "Pre-reasoning logprob at <answer_initial>, post-reasoning at <answer>. "
                             "Enables flip detection (initial != final label).")
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

    # ── Shared output file ────────────────────────────────────────────────────
    # bare-label mode → logprob_judgement_{lang}.json
    # --rationale mode → logprob_judgement_{lang}_rationale.json
    output_dir  = ROOT / "results_tmp" / "memerag_ext" / "logprob" / args.lang
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.chain:
        suffix = "_chain"
    elif args.rationale:
        suffix = "_rationale"
    else:
        suffix = ""
    output_path = output_dir / f"logprob_judgement_{args.lang}{suffix}.json"

    if args.chain:
        mode_str = "chain (answer_initial → rationale → answer → confidence)"
    elif args.rationale:
        mode_str = "rationale → answer → confidence"
    else:
        mode_str = "bare-label (step 0)"
    print(f"Mode: {mode_str}")
    print(f"Output: {output_path}\n")

    mode_val = "chain" if args.chain else ("rationale" if args.rationale else "bare-label")
    header = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "dataset_name": "memerag_ext",
        "lang":         args.lang,
        "mode":         mode_val,
        "models":       model_list,
        "data_file":    str(data_file),
        "n_samples":    len(candidate_data),
    }

    # Load existing records into shared dict {sample_id: record}
    all_records: dict[str, dict] = {}
    if not args.no_resume:
        all_records = load_existing(output_path)
        print(f"[resume] Loaded {len(all_records)} existing records from {output_path.name}\n")

    for model_name in model_list:
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        run_one_model(
            model_name       = model_name,
            candidate_data   = candidate_data,
            all_records      = all_records,
            output_path      = output_path,
            header           = header,
            no_resume        = args.no_resume,
            max_new_tokens   = args.max_new_tokens,
            checkpoint_every = args.checkpoint_every,
            use_rationale    = args.rationale,
            use_chain        = args.chain,
        )

    print(f"\nAll done. Final output: {output_path}")
    print(f"Total records: {len(all_records)}, models covered: {model_list}")


if __name__ == "__main__":
    main()
