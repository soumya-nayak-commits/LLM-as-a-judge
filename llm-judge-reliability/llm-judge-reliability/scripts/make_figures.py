"""Draw figures/*.png from results/summary/*.csv (run build_summary_tables.py first)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
S = lambda n: pd.read_csv(os.path.join(ROOT, "results", "summary", n))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, RED, GREY, GREEN, ORANGE = "#3b6ea5", "#c0392b", "#8c8c8c", "#2e8b57", "#d98324"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- Fig 1: agreement with humans ----
a = S("phase1_agreement.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, setting, title in [(axes[0], "all_300_with_ties", "All 300 pairs (3-way: A / B / tie)"),
                           (axes[1], "non_tie_243", "243 pairs where humans picked a side (forced choice)")]:
    d = a[a.setting == setting].sort_values("agreement")
    colors = [GREY if ("longer" in m or "model_b" in m) else (GREEN if "GPT-4" in m else BLUE)
              for m in d.method]
    ax.barh(d.method, d.agreement, color=colors)
    for y, (_, r_) in enumerate(d.iterrows()):
        right = r_.agreement
        if not np.isnan(r_.ci_lo):                       # draw a CI only where one was recorded
            ax.errorbar(r_.agreement, y, xerr=[[r_.agreement - r_.ci_lo], [r_.ci_hi - r_.agreement]],
                        fmt="none", ecolor="black", capsize=3)
            right = r_.ci_hi
        ax.text(right + 0.012, y, f"{r_.agreement:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("agreement with human majority label")
fig.suptitle("Fig 1. Qwen2.5-3B vs humans, with trivial baselines (grey) and GPT-4 (green). "
             "Black whiskers = 95% bootstrap CI by question (shown only where recorded)", fontsize=9)
save(fig, "fig1_agreement.png")

# ---- Fig 2: attacks and defenses ----
d4 = S("phase4_defenses.csv")
ad = S("phase4_adaptive_attacks.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
attacks = ["neutral_control", "direct_instruction", "fake_authority", "reworded_10_pooled"]
defs = [("plain", GREY), ("warning", ORANGE), ("sanitizer", GREEN)]
x = np.arange(len(attacks))
for k, (dname, col) in enumerate(defs):
    vals = [d4[(d4.defense == dname) & (d4.attack == at)].flip_rate.mean() for at in attacks]
    ax.bar(x + (k - 1) * 0.26, np.nan_to_num(vals), 0.26, label=dname, color=col)
    for xi, v in zip(x, vals):
        if np.isnan(v):                                   # condition was never run: say so, don't show 0
            ax.text(xi + (k - 1) * 0.26, 0.01, "not\nrun", ha="center", fontsize=6.5, color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(["neutral\ncontrol", "direct\ninstruction", "fake\nauthority", "10 reworded\n(pooled)"])
ax.set_ylabel("share of verdicts flipped to the rejected answer")
ax.set_title("Own attacks: sanitizer looks excellent...", fontsize=9)
ax.legend(title="defense", fontsize=8)
ax = axes[1]
x = np.arange(len(ad))
ax.bar(x - 0.2, ad.flip_no_sanitizer, 0.4, label="no sanitizer", color=GREY)
ax.bar(x + 0.2, ad.flip_sanitized, 0.4, label="sanitizer", color=RED)
ax.set_xticks(x)
ax.set_xticklabels(ad.attack, rotation=30, ha="right")
ax.set_title("...adaptive attacks written to evade it: it fails", fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(0, 0.6)
fig.suptitle("Fig 2. Share of verdicts flipped to the human-rejected answer (80 pairs). "
             "The warning barely helps; the sanitizer only beats attacks it was designed around", fontsize=9)
save(fig, "fig2_attacks_and_defenses.png")

# ---- Fig 3: audit ----
au = S("phase5_audit.csv")
d = au[au.flag == "inconsistent"].copy()
order = ["clean", "direct_instruction", "embedded_direct", "embedded_authority", "ALL_ATTACKED"]
d = d.set_index("condition").loc[order].reset_index()
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(d))
ax.bar(x - 0.2, d.catch_rate, 0.4, color=GREEN, label="catch rate (wrong verdicts flagged)")
ax.bar(x + 0.2, d.false_flag_rate, 0.4, color=RED, label="false-flag rate (correct verdicts flagged)")
ax.set_xticks(x)
ax.set_xticklabels(["clean", "direct\ninstruction", "embedded\ndirect", "embedded\nauthority", "all attacked\n(pooled)"])
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Fig 3. Order-inconsistency self-audit: works on clean pairs, fails under attack\n"
             "(useful only when the green bar is clearly above the red bar)", fontsize=9)
save(fig, "fig3_audit.png")

# ---- Fig 4: rubric trade-off ----
r = S("rubric_results.csv")
r = r[r.flip_rate_direct_instruction.notna()]
fig, ax = plt.subplots(figsize=(7, 4.6))
for _, row in r.iterrows():
    col = GREY if row.format == "pairwise" else BLUE
    xerr = None if np.isnan(row.agree_ci_lo) else [[row.agreement_243_non_tie - row.agree_ci_lo],
                                                   [row.agree_ci_hi - row.agreement_243_non_tie]]
    yerr = None if np.isnan(row.flip_ci_lo) else [[row.flip_rate_direct_instruction - row.flip_ci_lo],
                                                  [row.flip_ci_hi - row.flip_rate_direct_instruction]]
    ax.errorbar(row.agreement_243_non_tie, row.flip_rate_direct_instruction, xerr=xerr, yerr=yerr,
                fmt="o", color=col, capsize=2)
    ax.annotate(row.judge.replace("rubric v3: ", "").replace("rubric v1: ", "v1 "),
                (row.agreement_243_non_tie, row.flip_rate_direct_instruction),
                textcoords="offset points", xytext=(6, 5), fontsize=8)
ax.axvline(0.745, color=GREY, ls="--", lw=1)
ax.text(0.7465, 0.545, "always-longer\n0.745", fontsize=7, color=GREY)
ax.set_xlabel("agreement with humans (243 non-tie pairs)  -> better")
ax.set_ylabel("flip rate under direct-instruction attack  <- more robust")
ax.set_title("Fig 4. The best-agreeing criterion (helpfulness) is also among the easiest to manipulate\n"
             "Whiskers = 95% CI (by question); they overlap, so criteria cannot be ranked with confidence."
             " Plain pairwise: no CIs recorded.", fontsize=8)
ax.invert_yaxis()
save(fig, "fig4_rubric_tradeoff.png")

# ---- Fig 5: what moves the judge (scale) ----
b = S("phase2_bias.csv")
b = b[b.condition.isin(["pad_loser", "pad_winner", "all"]) & b.ci_lo.notna()]
a3 = S("phase3_attacks_80pairs.csv")
rows = []
for _, x_ in b.iterrows():
    nm = {"verbosity_generic_filler(266 chars)": "pad: generic filler",
          "verbosity_recap_filler(<=400 chars)": "pad: recap filler",
          "fluent_but_wrong(last number changed)": "one number made wrong"}[x_.test]
    cond = "" if x_.condition == "all" else f" ({x_.condition.replace('pad_', '')})"
    rows.append((nm + cond, x_.estimate, x_.ci_lo, x_.ci_hi, BLUE))
for _, x_ in a3.iterrows():
    nm = {"neutral_control": "attack: neutral control", "direct_instruction": "attack: direct instruction",
          "fake_authority": "attack: fake authority", "reworded_10_pooled": "attack: 10 reworded (pooled)"}[x_.attack]
    rows.append((nm, x_.mean_shift, x_.shift_ci_lo, x_.shift_ci_hi, RED))
fig, ax = plt.subplots(figsize=(8, 5))
for y, (nm, m, lo, hi, col) in enumerate(rows[::-1]):
    ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r_[0] for r_ in rows[::-1]], fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("mean margin shift toward the modified answer (log-odds units; 95% CI by question)")
ax.set_title("Fig 5. A wrong number moves the judge by about -1; a one-line instruction moves it by about +7",
             fontsize=9)
save(fig, "fig5_what_moves_the_judge.png")
