"""Phase 2: verbosity (padding) and fluent-but-wrong (one corrupted number).

Every measurement is a MARGIN SHIFT: how far the order-averaged margin moved toward the
side we modified (positive = toward it). `flipped_*` = verdict was NOT on that side before
and IS after.
"""
import re

import pandas as pd

from .._progress import tqdm
from ..corrupt import corrupted_row
from ..judge import margin
from ..rows import category, judged_index, other_side, padded_row, recap_row, side_col

FILLER = ("In summary, this answer addresses the question that was asked above. "
          "To restate, the main points have been covered as described. "
          "I hope this explanation is clear and helpful. Please let me know if "
          "you would like any further detail or clarification on any part of it.")


def verbosity(judge, sample, nt, kind: str = "generic") -> pd.DataFrame:
    """Pad the human-REJECTED answer ('pad_loser') or the PREFERRED one ('pad_winner').

    kind='generic': the same 266-char filler for every pair.
    kind='recap'  : the answer's own first sentences (up to 400 chars), so padding looks real.
    """
    out = []
    for _, r in tqdm(nt.iterrows(), total=len(nt)):
        row = sample.loc[int(r.idx)]
        win = r.human_label
        lose = other_side(win)
        for cond, side in [("pad_loser", lose), ("pad_winner", win)]:
            new_row = recap_row(row, side) if kind == "recap" else padded_row(row, side, FILLER)
            m_new = margin(judge, new_row)
            sgn = 1 if side == "model_a" else -1
            base_picks = (r.base_m > 0) == (side == "model_a")
            new_picks = (m_new > 0) == (side == "model_a")
            out.append({"idx": r.idx, "question_id": r.question_id, "cond": cond,
                        "shift": (m_new - r.base_m) * sgn,
                        "flipped_to_padded": (not base_picks) and new_picks})
    return pd.DataFrame(out)


def eligible_for_corruption(sample, nt) -> pd.DataFrame:
    """math / reasoning / extraction / STEM pairs whose preferred answer contains a number."""
    nt = nt.copy()
    nt["category"] = nt["question_id"].apply(category)

    def winner_text(r):
        row = sample.loc[int(r.idx)]
        conv = row[side_col(r.human_label)]
        return conv[judged_index(row.turn)]["content"]

    nt["winner_has_number"] = nt.apply(lambda r: bool(re.search(r"\d", winner_text(r))), axis=1)
    return nt[nt.category.isin(["math", "reasoning", "extraction", "stem"]) & nt.winner_has_number]


def fluent_wrong(judge, sample, elig) -> pd.DataFrame:
    """Corrupt one number in the human-preferred answer; did the judge notice?

    shift is signed so NEGATIVE means the judge liked the corrupted answer less (correct).
    No comparison group exists in this design (a known limit): the padding tests give a rough
    scale for what 'no content change' costs.
    """
    out = []
    for _, r in tqdm(elig.iterrows(), total=len(elig)):
        row = sample.loc[int(r.idx)]
        res = corrupted_row(row, r.human_label)
        if res is None:
            continue
        crow, old, new = res
        m_new = margin(judge, crow)
        side = r.human_label
        sgn = 1 if side == "model_a" else -1
        out.append({"idx": r.idx, "question_id": r.question_id, "category": r.category,
                    "old": old, "new": new, "shift": (m_new - r.base_m) * sgn,
                    "base_picks_winner": (r.base_m > 0) == (side == "model_a"),
                    "new_picks_winner": (m_new > 0) == (side == "model_a")})
    return pd.DataFrame(out)
