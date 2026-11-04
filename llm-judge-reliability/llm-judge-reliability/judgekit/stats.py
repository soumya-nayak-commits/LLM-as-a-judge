"""Statistics. Everything resamples by QUESTION, not by row.

Why: MT-Bench has only 80 questions, and many comparisons share a question, so rows are
not independent. Resampling rows would give confidence intervals that are too narrow.
"""
import numpy as np
import pandas as pd


def boot_mean(d: pd.DataFrame, col: str, n: int = 2000, seed: int = 0):
    """Mean of `col` with a 95% percentile bootstrap CI (resampling question_id groups).

    Returns (mean, lo, hi). Booleans are cast to float, so rates work too.
    """
    rng = np.random.default_rng(seed)
    g = {q: x[col].to_numpy(dtype=float) for q, x in d.groupby("question_id")}
    qs = np.array(list(g))
    vals = [np.concatenate([g[q] for q in rng.choice(qs, len(qs))]).mean() for _ in range(n)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(d[col].astype(float).mean()), float(lo), float(hi)


def boot_agreement(df: pd.DataFrame, col: str, ref: str = "human_label", **kw):
    """Share of rows where df[col] == df[ref], with a by-question bootstrap CI."""
    d = df.assign(_ok=(df[col] == df[ref]).astype(float))
    return boot_mean(d, "_ok", **kw)


def boot_paired_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, key: str = "idx", **kw):
    """Mean of (b[col] - a[col]) paired on `key`, bootstrapped by question.

    `a` and `b` must both carry `key` and `question_id`.
    """
    m = a[[key, "question_id", col]].merge(b[[key, col]], on=key, suffixes=("_a", "_b"))
    m["diff"] = m[f"{col}_b"] - m[f"{col}_a"]
    return boot_mean(m, "diff", **kw)
