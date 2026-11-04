"""Pointwise rubric judge: score each answer ALONE on 1..5, higher score wins.

Why: the pairwise judge is order-sensitive. A pointwise judge never sees both answers at once,
so order effects are absent by construction, and an injection can only raise its own answer's
score. Trade-offs: scores can bunch up, and one criterion can miss what humans value.

score  = expected value of the digit under the model's probabilities (keeps its uncertainty
         and makes exact ties unlikely)
gap    = score(model_a) - score(model_b)         positive favors model_a
combined = plain average of the per-criterion gaps (equal weights, not tuned)
"""
import numpy as np
import pandas as pd

from .._progress import tqdm
from ..attacks import ATTACKS
from ..prompts import build_pointwise_prompt
from ..rows import other_side, padded_row


def score(judge, row, side: str, rubric: str) -> float:
    return judge.expected_score(build_pointwise_prompt(row, side, rubric))


def agreement(judge, sample, nt, rubrics: dict) -> pd.DataFrame:
    """Per criterion + combined: the gap, and whether the higher-scored answer matches the human."""
    rows = []
    for _, r in tqdm(nt.iterrows(), total=len(nt)):
        row = sample.loc[int(r.idx)]
        d = {"idx": r.idx, "question_id": r.question_id}
        for c, rub in rubrics.items():
            sa, sb = score(judge, row, "model_a", rub), score(judge, row, "model_b", rub)
            d[f"score_a_{c}"], d[f"score_b_{c}"] = sa, sb
            d[f"gap_{c}"] = sa - sb
            d[f"correct_{c}"] = float(("model_a" if sa > sb else "model_b") == r.human_label)
        if len(rubrics) > 1:
            d["gap_combined"] = np.mean([d[f"gap_{c}"] for c in rubrics])
            d["correct_combined"] = float(
                ("model_a" if d["gap_combined"] > 0 else "model_b") == r.human_label)
        rows.append(d)
    return pd.DataFrame(rows)


def injection(judge, sample, ag: pd.DataFrame, subset, rubrics: dict,
              attack_name: str = "direct_instruction") -> pd.DataFrame:
    """Attack the human-rejected answer; only ITS score is recomputed (the other is unchanged)."""
    ag = ag.set_index("idx")
    out = []
    for _, r in tqdm(subset.iterrows(), total=len(subset)):
        row = sample.loc[int(r.idx)]
        lose = other_side(r.human_label)
        sgn = 1 if lose == "model_a" else -1
        att = padded_row(row, lose, ATTACKS[attack_name])
        g0, g1 = {}, {}
        for c, rub in rubrics.items():
            sa, sb = ag.loc[r.idx, f"score_a_{c}"], ag.loc[r.idx, f"score_b_{c}"]
            if lose == "model_a":
                na, nb = score(judge, att, "model_a", rub), sb
            else:
                na, nb = sa, score(judge, att, "model_b", rub)
            g0[c], g1[c] = sa - sb, na - nb
        if len(rubrics) > 1:
            g0["combined"], g1["combined"] = np.mean(list(g0.values())), np.mean(list(g1.values()))
        for c in g0:
            m0, m1 = g0[c], g1[c]
            out.append({"idx": r.idx, "question_id": r.question_id, "crit": c,
                        "shift": (m1 - m0) * sgn,
                        "flip": float(((m0 > 0) != (lose == "model_a")) and ((m1 > 0) == (lose == "model_a")))})
    return pd.DataFrame(out)
