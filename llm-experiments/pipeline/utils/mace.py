"""
MACE aggregation (Multi-Annotator Competence Estimation — Hovy et al. 2013).

Separated from aggregation.py because it requires crowd-kit + pandas.
"""
from __future__ import annotations

import math

try:
    import pandas as pd
    from crowdkit.aggregation import MACE as _CrowdMACE
    _CROWDKIT_AVAILABLE = True
except ImportError:
    _CROWDKIT_AVAILABLE = False


def run_mace(
    models: list[str],
    matrix: dict[str, list[int | None]],
    n_restarts: int = 10,
    n_iter: int = 50,
    method: str = "vb",       # "vb" = variational Bayes (default), "em" = standard EM
    smoothing: float = 0.1,
    default_noise: float = 0.5,
    alpha: float = 0.5,       # Beta prior on spamming probability
    beta: float = 0.5,        # Beta prior on spamming probability
    random_state: int = 0,
) -> tuple[list[int | None], dict[str, float], list[float]]:
    """Run MACE via crowd-kit.

    Returns
    -------
    labels           : hard prediction per sample (1/0/None)
    competence       : per-model non-spamming probability — higher = more reliable
    entropy_per_item : per-sample prediction entropy — higher = more uncertain
    """
    if not _CROWDKIT_AVAILABLE:
        raise ImportError("crowd-kit is required for MACE.  Run: pip install crowd-kit")

    n = len(next(iter(matrix.values())))

    # Build crowd-kit input: one row per (sample, model) observation
    rows = []
    for m in models:
        for i, lbl in enumerate(matrix[m]):
            if lbl is not None:
                rows.append({"task": i, "worker": m, "label": lbl})

    df = pd.DataFrame(rows)

    mace = _CrowdMACE(
        n_restarts=n_restarts,
        n_iter=n_iter,
        method=method,
        smoothing=smoothing,
        default_noise=default_noise,
        alpha=alpha,
        beta=beta,
        random_state=random_state,
    )
    pred_series = mace.fit_predict(df)

    # Hard labels (pred_series is a pandas Series indexed by task integer)
    labels: list[int | None] = [
        int(pred_series[i]) if i in pred_series.index else None
        for i in range(n)
    ]

    # Competence: spamming_ is shape (n_workers, 2)
    #   column 0 = spamming probability (higher = less reliable)
    #   column 1 = non-spamming probability (higher = more reliable)
    # Workers are ordered by first appearance in df["worker"], which matches `models` order.
    worker_to_idx = {m: i for i, m in enumerate(df["worker"].unique())}
    spamming = mace.spamming_
    competence: dict[str, float] = {
        m: float(spamming[worker_to_idx[m], 1]) if m in worker_to_idx else float("nan")
        for m in models
    }

    # Per-item entropy from probas_ (DataFrame: columns = label values, index = task int)
    probas = mace.probas_
    entropy_per_item: list[float] = []
    for i in range(n):
        if i in probas.index:
            probs = probas.loc[i].values
            h = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            entropy_per_item.append(round(h, 6))
        else:
            entropy_per_item.append(float("nan"))

    return labels, competence, entropy_per_item
