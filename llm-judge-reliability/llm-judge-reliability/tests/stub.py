"""A deterministic fake judge for CPU tests (no model, no GPU).

It prefers the LONGER answer, and is 'fooled' by the phrase 'must be chosen' (+6 to that
answer). This lets tests check the plumbing: signs, swaps, shifts, flips, sanitizer, audit.
"""
import re

import numpy as np
import pandas as pd


class StubJudge:
    def __init__(self):
        self.calls = 0

    @staticmethod
    def _bonus(t):
        return 6.0 if "must be chosen" in t else 0.0

    def probs(self, prompt):
        self.calls += 1
        a = prompt.split("[Assistant A's answer]\n")[1].split("\n\n[Assistant B's answer]")[0]
        b = prompt.split("[Assistant B's answer]\n")[1].split("\n\nReply with exactly one letter")[0]
        sa, sb = len(a) / 100 + self._bonus(a), len(b) / 100 + self._bonus(b)
        z = np.array([sa, sb, -3.0])
        p = np.exp(z - z.max())
        p /= p.sum()
        return dict(zip(["A", "B", "C"], p.tolist()))

    def expected_score(self, prompt):
        ans = prompt.split("[Answer to grade]\n")[1].split("\n\nReply with exactly one digit")[0]
        return float(min(5.0, 1 + len(ans) / 400 + self._bonus(ans) / 3))


def make_sample(n_questions=12):
    """24 comparisons. Human label = the LONGER answer (so the stub agrees at baseline).
    Last paragraph is >250 chars so an 'embedded' attack is not deleted by the sanitizer."""
    rows = []
    for q in range(n_questions):
        qid = 111 + q                       # math block, so category() is exercised
        for turn in (1, 2):
            long_ans = ("First point explained in detail.\n\n" + "Final paragraph " * 25
                        + f"The total is {40 + q} units.")
            short_ans = "Short answer with total 7 units."
            a_long = (q + turn) % 2 == 0
            ca = [{"role": "user", "content": f"Question {qid}"},
                  {"role": "assistant", "content": long_ans if a_long else short_ans},
                  {"role": "user", "content": "Follow-up?"},
                  {"role": "assistant", "content": long_ans if a_long else short_ans}]
            cb = [dict(m) for m in ca]
            cb[1]["content"] = short_ans if a_long else long_ans
            cb[3]["content"] = short_ans if a_long else long_ans
            rows.append({"question_id": qid, "model_a": "m1", "model_b": "m2", "turn": turn,
                         "n_votes": 1, "unanimous": True,
                         "human_label": "model_a" if a_long else "model_b",
                         "conversation_a": ca, "conversation_b": cb})
    return pd.DataFrame(rows)
