"""Phases 3-4: attack the judge, then defend it. ONE function runs every condition so the
comparison between conditions is fair (same pairs, same attack texts, same metric).

Metrics (per pair, attack applied to the answer humans REJECTED)
  shift   : (attacked margin - clean margin), signed so POSITIVE = moved toward the rejected answer
  flipped : verdict was NOT on the rejected answer before the attack and IS after
  removed : (sanitizer only) the attack text was completely deleted, so the judge saw a clean answer
"""
import numpy as np
import pandas as pd

from .._progress import tqdm
from ..attacks import DETECT, apply_attack, as_modes, ATTACKS, flagged, sanitize_row
from ..judge import margin
from ..prompts import build_prompt
from ..rows import judged_answers, judged_index, other_side, side_col


def attack_grid(judge, sample, subset, attacks: dict, prompt_fn=build_prompt,
                sanitize: bool = False) -> pd.DataFrame:
    """attacks: {name: (mode, text)}, mode in {'para','embed'}. subset needs idx/human_label/question_id.

    sanitize=True: the sanitizer runs on the attacked row (and on the clean baseline) before the
    judge sees it. If sanitizing makes the attacked row identical to the clean sanitized row, no
    model call is needed and the shift is exactly 0.
    """
    out = []
    for _, r in tqdm(subset.iterrows(), total=len(subset)):
        row = sample.loc[int(r.idx)]
        lose = other_side(r.human_label)
        sgn = 1 if lose == "model_a" else -1
        clean_row = sanitize_row(row) if sanitize else row
        base = margin(judge, clean_row, prompt_fn)
        base_picks = (base > 0) == (lose == "model_a")
        for name, (mode, text) in attacks.items():
            att = apply_attack(row, lose, mode, text)
            removed = False
            if sanitize:
                att = sanitize_row(att)
                removed = judged_answers(att) == judged_answers(clean_row)
            m_new = base if removed else margin(judge, att, prompt_fn)
            new_picks = (m_new > 0) == (lose == "model_a")
            out.append({"idx": r.idx, "question_id": r.question_id, "attack": name,
                        "removed": removed, "shift": (m_new - base) * sgn,
                        "flipped_to_attacked": (not base_picks) and new_picks})
    return pd.DataFrame(out)


def regex_detection(sample, nt, attacks_text: dict) -> dict:
    """Detection rate of the deterministic regex on attacked answers, and its false positives."""
    det = {}
    for name, text in attacks_text.items():
        hits = []
        for _, r in nt.iterrows():
            row = sample.loc[int(r.idx)]
            a = apply_attack(row, other_side(r.human_label), "para", text)
            col = side_col(other_side(r.human_label))
            hits.append(flagged(a[col][judged_index(row.turn)]["content"]))
        det[name] = float(np.mean(hits))
    clean = []
    for _, row in sample.iterrows():
        clean += [flagged(t) for t in judged_answers(row)]
    det["_false_positive_rate_clean_answers"] = float(np.mean(clean))
    return det


def sanitizer_false_positives(sample, subset) -> int:
    """How many CLEAN pairs does the sanitizer alter? (the cost of the defense)"""
    n = 0
    for _, r in subset.iterrows():
        row = sample.loc[int(r.idx)]
        n += judged_answers(sanitize_row(row)) != judged_answers(row)
    return int(n)
