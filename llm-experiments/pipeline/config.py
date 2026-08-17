# =============================================================================
# Pipeline configuration — edit this file only
# =============================================================================

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET = "memerag"   # "memerag" | "qags"

# ── Language / subset ─────────────────────────────────────────────────────────
# memerag: en | de | es | fr | hi
# qags   : cnndm | xsum
LANG = "en"

# ── Proposer models (judges that produce per-sample labels) ───────────────────
PROPOSERS = [
    "MiniMax-M2.7",
    "GPT-OSS-120b",
    "Qwen3.6-35B",
    "gemini-2.5-flash",
    "meta/llama-3.3-70b-instruct-maas",
    "google/gemma-4-26b-a4b-it-maas",
    "Apertus-8B",               # outlier — excluded from aggregator candidate pool
    # "gemini-2.5-flash-lite",      # in IGNORE_MODELS — excluded from aggregation
    # "gpt-5.4-mini",               # in IGNORE_MODELS
    # "gpt-5.4-mini-2026-03-17",    # in IGNORE_MODELS
    # "Qwen3.5-122B",               # connection issues
]

# Models to skip in metrics / display (still in JSON, just excluded from analysis)
IGNORE_MODELS: set[str] = {
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
    "gemini-2.5-flash-lite",
}

# ── Algorithmic aggregation (no LLM call) ─────────────────────────────────────
# Runs on proposer votes — all selected algos run in one pass, each saves its own output
# Available: "majority" | "owi" | "isp" | "iwmv" | "ds" | "mace"
#   isp  = real Inverse Surprising Popularity (Algorithm 2, arXiv:2510.01499)
#   iwmv = Iterative Weighted Majority Vote (what was previously mislabeled as ISP)
#   owi  = OW-I (logit-weighted vote using real ISP pseudo-labels as reference)
# Set to [] to skip
AGGREGATION_ALGOS = ["majority", "owi", "isp", "iwmv", "ds", "mace"]

# ── Aggregator LLM selection (makes API call) ─────────────────────────────────
# Picks one proposer per sample to act as aggregator LLM (sees all other outputs)
#   "random"     — pick a random proposer each sample
#   "fixed"      — always use AGGREGATOR_MODEL
#   "ds_rank"    — pick the Dawid-Skene highest-ranked proposer
#   "best_bacc"  — pick the proposer with highest balanced accuracy
#   "worse_bacc" — pick the proposer with lowest balanced accuracy (ablation)
#   None         — skip LLM aggregation entirely
AGGREGATOR_LLM   = "random"
# AGGREGATOR_MODEL = "MiniMax-M2.7"   # only used when AGGREGATOR_LLM = "fixed"

# ── Exlude model ─────────────────────────────────────────────────────────────
# Exclude a model from both proposer and aggregator roles
#   None        — no exclusion (full panel)
#   "ModelName" — exclude that specific model
#   "all"       — exhaustive LOO (runs once per proposer, excluding each in turn)
EXCLUDE_MODEL = None

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED = 42
CHECKPOINT_EVERY = 5
