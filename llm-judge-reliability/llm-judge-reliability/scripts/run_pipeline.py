"""Run the whole study on a GPU (Colab / Kaggle T4). Results go to results/raw/*.csv.

    python scripts/run_pipeline.py --phase all
    python scripts/run_pipeline.py --phase 3 --model Qwen/Qwen2.5-3B-Instruct

Each phase saves its CSV as soon as it finishes, and phase 1 is re-used (loaded from disk) by
later phases, so a disconnect only costs you the phase that was running.
Runtime on a free T4 with Qwen2.5-3B-Instruct: rough estimate 1 to 1.5 hours end to end
(extrapolated from ~3 forward passes/s seen in the notebook; not measured for this script).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from judgekit import data  # noqa: E402
from judgekit.attacks import ADAPTIVE, ATTACKS, REWORDED, as_modes  # noqa: E402
from judgekit.experiments import (phase1_agreement as p1, phase2_bias as p2,  # noqa: E402
                                  phase3_attacks as p3, phase5_audit as p5, pointwise_rubric as pw)
from judgekit.prompts import ACC_RUBRIC_V1, RUBRICS, build_prompt, build_prompt_defended  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")


def save(df, name):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name)
    df.to_csv(path, index=False)
    print("saved", path, len(df), "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", help="1,2,3,4,5,rubric or all")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    want = {"1", "2", "3", "4", "5", "rubric"} if a.phase == "all" else set(a.phase.split(","))

    ds = data.load_dataset_dict()
    pairs = data.collapse_votes(ds["human"].to_pandas())
    clean = data.drop_splits(pairs)
    sample = data.draw_sample(clean, n=a.n, seed=42)
    print(f"{len(pairs)} comparisons -> {len(clean)} usable -> sample {len(sample)}")

    from judgekit.judge import HFJudge
    judge = HFJudge(a.model)

    p1_path = os.path.join(RAW, "phase1_results.csv")
    if os.path.exists(p1_path):
        res = pd.read_csv(p1_path)
    else:
        res = p1.run_pairwise(judge, sample)
        res = p1.add_derived(res, sample, data.gpt4_labels(sample, ds["gpt4_pair"].to_pandas()))
        save(res, "phase1_results.csv")
    nt = p1.non_tie(res)                       # 243 non-tie pairs (+ base_m)
    sub = nt.sample(80, random_state=0)        # the 80-pair subset used for expensive tests

    if "1" in want:
        save(p1.agreement_table(res), "phase1_agreement.csv")
        save(pd.DataFrame([p1.order_sensitivity(res)]), "phase1_order_sensitivity.csv")

    if "2" in want:
        save(p2.verbosity(judge, sample, nt, "generic"), "phase2_verbosity_generic.csv")
        save(p2.verbosity(judge, sample, nt, "recap"), "phase2_verbosity_recap.csv")
        elig = p2.eligible_for_corruption(sample, nt)
        save(p2.fluent_wrong(judge, sample, elig), "phase2_fluent_wrong.csv")

    if "3" in want:
        save(p3.attack_grid(judge, sample, nt, as_modes(ATTACKS)), "phase3_attacks_243.csv")
        rw = {f"reworded_{k}": ("para", t) for k, t in enumerate(REWORDED)}
        save(p3.attack_grid(judge, sample, sub, rw), "phase3_reworded_80.csv")
        det = p3.regex_detection(sample, nt, {**ATTACKS, **{f"rw{k}": t for k, t in enumerate(REWORDED)}})
        save(pd.DataFrame([det]), "phase3_regex_detection.csv")

    if "4" in want:
        atk = {**as_modes(ATTACKS), **{f"reworded_{k}": ("para", t) for k, t in enumerate(REWORDED)}}
        save(p3.attack_grid(judge, sample, sub, as_modes(ATTACKS)).assign(defense="plain"),
             "phase4_plain_80.csv")
        save(p3.attack_grid(judge, sample, sub, as_modes(ATTACKS), build_prompt_defended)
             .assign(defense="warning"), "phase4_warning_80.csv")
        save(p3.attack_grid(judge, sample, sub, atk, sanitize=True).assign(defense="sanitizer"),
             "phase4_sanitizer_80.csv")
        print("sanitizer false positives on clean pairs:", p3.sanitizer_false_positives(sample, sub), "/ 80")
        save(p3.attack_grid(judge, sample, sub, ADAPTIVE).assign(defense="none_adaptive"),
             "phase4_adaptive_raw_80.csv")
        save(p3.attack_grid(judge, sample, sub, ADAPTIVE, sanitize=True).assign(defense="sanitizer_adaptive"),
             "phase4_adaptive_sanitized_80.csv")

    if "5" in want:
        au = p5.run_audit(judge, sample, sub)
        save(au, "phase5_audit_rows.csv")
        save(p5.audit_report(au), "phase5_audit_report.csv")

    if "rubric" in want:
        v1 = {"accuracy_v1": ACC_RUBRIC_V1}
        ag1 = pw.agreement(judge, sample, nt, v1)
        save(ag1, "rubric_v1_agreement.csv")
        save(pw.injection(judge, sample, ag1, sub, v1), "rubric_v1_injection.csv")
        ag3 = pw.agreement(judge, sample, nt, RUBRICS)
        save(ag3, "rubric_v3_agreement.csv")
        save(pw.injection(judge, sample, ag3, sub, RUBRICS), "rubric_v3_injection.csv")


if __name__ == "__main__":
    main()
