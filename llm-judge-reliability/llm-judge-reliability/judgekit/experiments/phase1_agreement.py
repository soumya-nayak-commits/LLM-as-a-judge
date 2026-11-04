"""Phase 1: judge vs human agreement, baselines, order sensitivity.

Terms
  with ties     : 3-way label (model_a / model_b / tie) on ALL pairs
  forced choice : tie is not allowed, take the larger of pA and pB; scored on pairs where
                  the human did not say tie (non-tie rows)
  order-avg     : add probabilities from both orders, take the argmax
  hard flip     : picked model_a in one order and model_b in the other
  soft flip     : verdict changed but one of the two verdicts was a tie
"""
import numpy as np
import pandas as pd

from .._progress import tqdm
from ..judge import lg
from ..prompts import build_prompt
from ..rows import longer_label, swap_row
from ..stats import boot_agreement

LETTER_TO_LABEL = {"A": "model_a", "B": "model_b", "C": "tie"}
SWAP_MAP = {"A": "model_b", "B": "model_a", "C": "tie"}   # swapped-run letter -> original frame


def run_pairwise(judge, sample, prompt_fn=build_prompt) -> pd.DataFrame:
    rows = []
    for i, row in tqdm(sample.iterrows(), total=len(sample)):
        p = judge.probs(prompt_fn(row))
        sp = judge.probs(prompt_fn(swap_row(row)))
        rows.append({
            "idx": i, "question_id": row.question_id, "turn": row.turn,
            "human_label": row.human_label,
            "judge_label": LETTER_TO_LABEL[max(p, key=p.get)],
            "pA": p["A"], "pB": p["B"], "pC": p["C"],
            "swap_label": SWAP_MAP[max(sp, key=sp.get)],
            "sw_pA": sp["A"], "sw_pB": sp["B"], "sw_pC": sp["C"],
        })
    return pd.DataFrame(rows)


def add_derived(res: pd.DataFrame, sample: pd.DataFrame, gpt4: pd.DataFrame = None):
    res = res.copy()
    score = pd.DataFrame({"model_a": res.pA + res.sw_pB,
                          "model_b": res.pB + res.sw_pA,
                          "tie": res.pC + res.sw_pC})
    res["avg_label"] = score.idxmax(axis=1)
    res["forced_label"] = np.where(res.pA > res.pB, "model_a", "model_b")
    res["forced_swap"] = np.where(res.sw_pA > res.sw_pB, "model_b", "model_a")
    res["forced_avg"] = np.where(res.pA + res.sw_pB > res.pB + res.sw_pA, "model_a", "model_b")
    res["longer"] = [longer_label(sample.loc[i]) for i in res["idx"]]
    if gpt4 is not None:
        res = res.drop(columns=["gpt4_label"], errors="ignore").merge(gpt4, on="idx", how="left")
    return res


def non_tie(res: pd.DataFrame) -> pd.DataFrame:
    """Rows where the human picked a side, plus base_m (order-averaged clipped log-margin)."""
    nt = res[res.human_label != "tie"].copy()
    nt["base_m"] = 0.5 * ((lg(nt.pA) - lg(nt.pB)) + (lg(nt.sw_pB) - lg(nt.sw_pA)))
    return nt


def agreement_table(res: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Agreement with humans (+ by-question 95% CI) for every judge variant and baseline."""
    rows = []

    def add(setting, method, df, col):
        a, lo, hi = boot_agreement(df, col, seed=seed)
        rows.append({"setting": setting, "method": method, "n": len(df),
                     "agreement": a, "ci_lo": lo, "ci_hi": hi})

    for name, col in [("Qwen original", "judge_label"), ("Qwen swapped", "swap_label"),
                      ("Qwen order-avg", "avg_label"), ("Always longer", "longer")]:
        add("all_pairs_with_ties", name, res, col)
    if "gpt4_label" in res:
        add("all_pairs_with_ties", "GPT-4", res, "gpt4_label")
    for name, col in [("Qwen forced", "forced_label"), ("Qwen forced swapped", "forced_swap"),
                      ("Qwen forced order-avg", "forced_avg"), ("Always longer", "longer")]:
        add("non_tie_pairs", name, res[res.human_label != "tie"], col)
    n = res[res.human_label != "tie"]
    rows.append({"setting": "non_tie_pairs", "method": "Always model_b", "n": len(n),
                 "agreement": float((n.human_label == "model_b").mean()),
                 "ci_lo": np.nan, "ci_hi": np.nan})
    rows.append({"setting": "all_pairs_with_ties", "method": "Always model_b", "n": len(res),
                 "agreement": float((res.human_label == "model_b").mean()),
                 "ci_lo": np.nan, "ci_hi": np.nan})
    return pd.DataFrame(rows)


def order_sensitivity(res: pd.DataFrame) -> dict:
    both = res.judge_label.ne("tie") & res.swap_label.ne("tie")
    hard = both & (res.judge_label != res.swap_label)
    soft = (res.judge_label != res.swap_label) & ~hard
    nt = res[res.human_label != "tie"]
    return {
        "consistent": int((res.judge_label == res.swap_label).sum()),
        "hard_flips": int(hard.sum()), "soft_flips": int(soft.sum()),
        "forced_choice_consistency": float((res.forced_label == res.forced_swap).mean()),
        "qwen_tie_rate": float((res.judge_label == "tie").mean()),
        "human_tie_rate": float((res.human_label == "tie").mean()),
        "qwen_forced_picks_longer": float((nt.forced_label == nt.longer).mean()),
        "human_picks_longer": float((nt.human_label == nt.longer).mean()),
    }
