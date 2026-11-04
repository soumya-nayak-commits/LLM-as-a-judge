"""Data: MT-Bench human judgments -> one label per comparison -> a fixed sample.

Source: lmsys/mt_bench_human_judgments (Hugging Face). Two tables:
  * "human"      one row per human VOTE (3,355 rows)   <- ground truth we use
  * "gpt4_pair"  GPT-4 verdicts on the same kind of pairs <- reference point only
A "comparison" = (question_id, model_a, model_b, turn). Turn 1 scores the first reply,
turn 2 scores the reply to the follow-up question.
"""
import pandas as pd

DATASET_ID = "lmsys/mt_bench_human_judgments"
KEY = ["question_id", "model_a", "model_b", "turn"]


def load_dataset_dict():
    from datasets import load_dataset
    return load_dataset(DATASET_ID)


def collapse_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Votes -> one row per comparison, using a STRICT majority.

    If two labels tie for first place (e.g. one vote A, one vote B) the label is "split":
    there is no trustworthy human answer, so these rows are dropped later and only counted.
    """
    rows = []
    for key, grp in df.groupby(KEY):
        votes = grp["winner"].value_counts()
        top = votes.iloc[0]
        strict_majority = (votes == top).sum() == 1
        rows.append({
            "question_id": key[0], "model_a": key[1], "model_b": key[2], "turn": key[3],
            "n_votes": len(grp),
            "human_label": votes.index[0] if strict_majority else "split",
            "unanimous": votes.iloc[0] == len(grp),
            "conversation_a": grp["conversation_a"].iloc[0],
            "conversation_b": grp["conversation_b"].iloc[0],
        })
    return pd.DataFrame(rows)


def drop_splits(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs[pairs["human_label"] != "split"].reset_index(drop=True)


def draw_sample(clean: pd.DataFrame, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Fixed-seed sample. The index (0..n-1) is the `idx` used everywhere downstream."""
    return clean.sample(n=n, random_state=seed).reset_index(drop=True)


def pair_key(df: pd.DataFrame):
    """Order-free key: GPT-4's table may list a pair with model_a/model_b swapped."""
    return df.apply(lambda r: (r.question_id, r.turn, *sorted([r.model_a, r.model_b])), axis=1)


def gpt4_labels(sample: pd.DataFrame, gpt4_table: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[idx, gpt4_label] in OUR model_a/model_b frame.

    "tie (inconsistent)" (GPT-4 changed its mind when order was swapped) is merged into "tie".
    """
    gpt = gpt4_table[["question_id", "model_a", "model_b", "turn", "winner"]].rename(
        columns={"winner": "gpt4_raw"})
    gpt["key"] = pair_key(gpt)
    gpt = gpt.drop_duplicates("key")
    s = sample[["question_id", "model_a", "model_b", "turn"]].copy()
    s["key"] = pair_key(s)
    s["idx"] = s.index
    m = s.merge(gpt[["key", "model_a", "gpt4_raw"]].rename(columns={"model_a": "gpt_model_a"}),
                on="key", how="left")
    flip = {"model_a": "model_b", "model_b": "model_a", "tie": "tie",
            "tie (inconsistent)": "tie"}
    same_frame = m["gpt_model_a"] == m["model_a"]
    m["gpt4_label"] = m["gpt4_raw"].where(same_frame, m["gpt4_raw"].map(flip))
    m["gpt4_label"] = m["gpt4_label"].replace({"tie (inconsistent)": "tie"})
    return m[["idx", "gpt4_label"]]
