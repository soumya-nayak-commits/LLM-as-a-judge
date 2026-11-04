"""'Fluent but wrong': change ONE number in the human-preferred answer.

The corruption must be a real error, so we skip list markers ("1.") and step labels
("Step 3"), handle thousands separators, and keep the decimal precision.
"""
import re

from .rows import judged_index, side_col


def corrupt_number(text: str):
    """Change the LAST real number. Returns (new_text, old, new) or None if there is none."""
    cands = []
    for m in re.finditer(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        is_list_marker = (m.start() == line_start and text[m.end():m.end() + 1] in ".)")
        is_step = text[max(0, m.start() - 6):m.start()].lower().endswith(("step ", "step"))
        if not (is_list_marker or is_step):
            cands.append(m)
    if not cands:
        return None
    m = cands[-1]
    old = m.group()
    clean_old = old.replace(",", "")
    if "." in clean_old:
        dec = len(clean_old.split(".")[1])
        new = f"{float(clean_old) * 1.3 + 0.5:.{dec}f}"
    else:
        n = int(clean_old)
        new = str(n + 2 + n // 5)
    if "," in old:
        new = f"{int(float(new)):,}" if "." not in clean_old else new
    return text[:m.start()] + new + text[m.end():], old, new


def corrupted_row(row, side: str):
    """Copy of `row` with the judged answer on `side` corrupted; None if it has no number."""
    col = side_col(side)
    j = judged_index(row.turn)
    res = corrupt_number(row[col][j]["content"])
    if res is None:
        return None
    new_text, old, new = res
    conv = [dict(m) for m in row[col]]
    conv[j]["content"] = new_text
    r = row.copy()
    r[col] = conv
    return r, old, new
