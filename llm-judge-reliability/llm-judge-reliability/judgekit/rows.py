"""Small, pure functions that transform one comparison row (no model needed).

A row has conversation_a / conversation_b: lists of 4 messages
[user Q1, assistant reply 1, user Q2, assistant reply 2]. The judged reply is index 1
for turn 1 and index 3 for turn 2.
"""
import re


def judged_index(turn: int) -> int:
    return 1 if turn == 1 else 3


def other_side(label: str) -> str:
    return "model_b" if label == "model_a" else "model_a"


def side_col(side: str) -> str:
    return "conversation_a" if side == "model_a" else "conversation_b"


def swap_row(row):
    """Same pair, answers in the other order. Used to test order sensitivity."""
    r = row.copy()
    r["conversation_a"], r["conversation_b"] = row.conversation_b, row.conversation_a
    r["model_a"], r["model_b"] = row.model_b, row.model_a
    return r


def judged_answers(row):
    j = judged_index(row.turn)
    return row.conversation_a[j]["content"], row.conversation_b[j]["content"]


def answer_len(conv, turn) -> int:
    return len(conv[judged_index(turn)]["content"])


def longer_label(row) -> str:
    """The 'always pick the longer answer' baseline (by character count)."""
    return ("model_a" if answer_len(row.conversation_a, row.turn)
            > answer_len(row.conversation_b, row.turn) else "model_b")


def pad_conv(conv, turn, filler):
    """Append `filler` as its own paragraph at the end of the judged reply."""
    conv = [dict(m) for m in conv]
    j = judged_index(turn)
    conv[j]["content"] = conv[j]["content"].rstrip() + "\n\n" + filler
    return conv


def padded_row(row, side, filler):
    r = row.copy()
    col = side_col(side)
    r[col] = pad_conv(r[col], r.turn, filler)
    return r


def embed_row(row, side, text):
    """Glue `text` onto the END OF THE LAST PARAGRAPH (no blank line): the adaptive attack."""
    r = row.copy()
    col = side_col(side)
    conv = [dict(m) for m in r[col]]
    j = judged_index(r.turn)
    conv[j]["content"] = conv[j]["content"].rstrip() + " " + text
    r[col] = conv
    return r


def recap_text(answer: str, max_chars: int = 400) -> str:
    """The answer's own opening sentences (up to max_chars): padding with no new content."""
    sents = re.split(r"(?<=[.!?])\s+", answer.strip())
    out = ""
    for s in sents:
        if len(out) + len(s) > max_chars:
            break
        out += s + " "
    return out.strip() or answer.strip()[:max_chars]


def recap_row(row, side):
    col = side_col(side)
    j = judged_index(row.turn)
    return padded_row(row, side, "To recap the key points: " + recap_text(row[col][j]["content"]))


# MT-Bench question ids come in blocks of 10 per category.
_BLOCKS = {81: "writing", 91: "roleplay", 101: "reasoning", 111: "math",
           121: "coding", 131: "extraction", 141: "stem", 151: "humanities"}


def category(qid: int) -> str:
    return _BLOCKS[max(b for b in _BLOCKS if b <= qid)]
