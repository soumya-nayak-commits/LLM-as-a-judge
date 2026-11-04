# Methods and metrics: what each number means

Read this with the README. Each section: **what it is**, **how it is computed**, **why this choice**,
**the strongest objection and the honest answer**.

---
## 0. Vocabulary
- **Comparison**: one question + one pair of models + one turn. Turn 1 judges the first reply; turn 2 judges the reply to the follow-up (with the first exchange as context).
- **Human label**: the strict-majority vote among the humans who judged that comparison. If two labels tie for first place it is `split` and the comparison is dropped (199 of 2,396). 68% of kept labels rest on one vote.
- **Judge**: Qwen2.5-3B-Instruct. It never writes a verdict; we read the probability it gives to the next token.
- **Rejected / preferred answer**: the answer the human label says lost / won. Attacks go on the *rejected* one.

## 1. Reading verdicts from probabilities
**Compute:** run one forward pass on the chat-formatted prompt, take the logits of the tokens `A`, `B`, `C` at the last position, softmax over those three.
**Why:** no parsing failures, no temperature randomness, and the probabilities give a confidence, so "borderline" has a definition.
**Objection:** "Real judges generate reasoning first." **Answer:** true, chain-of-thought can change results; we did not test it. We measure the cheapest, most deterministic reading and say so.

## 2. Agreement
- **With ties (3-way):** judge label (`model_a` / `model_b` / `tie`) equals human label, over all 300 pairs.
- **Forced choice:** tie disallowed (take the larger of pA, pB), scored on the 243 pairs where the human picked a side.
**Why both:** the judge says tie 40% of the time vs 19% for humans, which dominates the 3-way score. Forced choice isolates "can it pick the better answer when it must?"
**Objection:** "Forced choice isn't how the judge ran." **Answer:** correct, it is a *counterfactual* that diagnoses the tie problem; the 3-way number is the realistic one and is reported first.

## 3. Baselines (always-longer, always-model_b, random)
**Why:** agreement is meaningless without a floor. Always-longer reaches **0.745** on non-tie pairs; forced Qwen reaches **0.700**. If you only report "0.70", it sounds good; against the baseline it is not.
**Objection:** "Humans also like longer answers, so this baseline is unfair." **Answer:** that is the point. It shows how much of the apparent skill is a length proxy.

## 4. Confidence intervals: bootstrap by question
**Compute:** resample the *question ids* with replacement (2,000 times), pull in all rows of each chosen question, recompute the statistic, take the 2.5th and 97.5th percentiles.
**Why:** MT-Bench has 80 questions; rows from one question are correlated. Resampling rows would make the CI too narrow and overstate certainty.
**Objection:** "With ~80 clusters the bootstrap is itself noisy." **Answer:** agreed; that is why we call overlapping intervals "no evidence of a difference" rather than "equal", and avoid ranking small gaps.

## 5. Order sensitivity (the swap test)
**Compute:** judge each pair twice, answers in both orders; translate the swapped verdict back to the original frame.
- **Consistent**: same verdict both times. **Hard flip**: `model_a` in one order, `model_b` in the other. **Soft flip**: changed, but one verdict was `tie`.
- **Slot preference:** among hard flips, does the judge pick the first-shown (or second-shown) answer in both runs? Pilot: no clear preference.
**Why we say "order-sensitive", not "position-biased":** position bias means a systematic preference for a slot; we found instability without a consistent slot.
**Objection:** "Slot preference was checked on 100 pairs only." **Answer:** yes, stated as such.

## 6. Margin (a single signed number per pair)
`margin = 0.5 * [ (log pA - log pB) + (log sw_pB - log sw_pA) ]`
- First term: original order; second: swapped order, with letters reversed so the sign is back in the original frame. Positive = favors `model_a`.
- **Why average both orders:** cancels most order effects so a shift is attributable to our modification.
- **Why clip probabilities at 1e-6:** float16 can underflow to exactly 0, giving `log 0 = -inf` and `nan`. Clipping caps margins at about +-14. A shift of +7 is large but not at the cap.
- **The sign is a convention** we chose by the order of subtraction.

## 7. Shift and flip rate
- **Shift** = (margin after modification - margin before) x (+1 if the modified answer is `model_a`, else -1). Positive always means "moved toward the modified answer".
- **Flip** = the order-averaged verdict was **not** on the modified answer before and **is** after.
**Why both:** shift is continuous and sensitive; flip is the practical consequence ("did the verdict actually change?").
**Objection:** "Pairs where the judge already favored the rejected answer can't flip, so flip rate understates vulnerability." **Answer:** correct; it is a lower bound on exposure, noted in the analysis.

## 8. Why attack the rejected answer, and why a neutral control
**Why the rejected answer:** it is the case where manipulation causes real harm (a bad answer wins). Attacking the winner can only reinforce a correct verdict.
**Neutral control:** harmless bracketed text of similar length. Any attack effect *beyond* the control is due to what the text **says**, not to the answer getting longer or having a bracket.
**All attack texts are position-agnostic** ("this answer", never "Assistant A"), so success cannot come from exploiting letter positions.

## 9. Regex detector vs paraphrases
**Result:** 100% on the wordings it was written to match, 0/10 on paraphrases.
**What it shows:** a phrase detector recognises words, not meaning. The 100% is **circular by construction** and should never be quoted alone. The paraphrases are author-written, so 0/10 is also not a general detection rate.

## 10. Defenses, and why we report their cost
- **Warning** (one added instruction): same prompt otherwise, so any change is due to the warning. Result: negligible.
- **Sanitizer** (delete paragraphs that are short and contain cue words): reported with its **false positives** (10/80 clean pairs altered). A defense that damages normal inputs is not free.
- **Circularity caveat:** the cue list was written after seeing the attacks. Adaptive attacks (glued onto a long paragraph / no cue words) test that. The no-cue attacks are weak even without a defense, so "evaded the sanitizer" does not mean "dangerous".

## 11. Self-audit: catch rate and false-flag rate
- A verdict is **wrong** if it favors the human-rejected answer.
- **Flag** = a rule using only the judge's own behavior (order-inconsistent, or |margin| below a threshold).
- **Catch rate** = P(flagged | wrong). **False-flag rate** = P(flagged | correct). A flag is useful only if catch is clearly above false-flag. Always report both, plus the share flagged (the review workload).
**Why not just accuracy of the flag:** the classes are imbalanced and the cost is asymmetric (a missed wrong verdict vs wasted review).
**Objection:** "'Wrong' includes baseline mistakes unrelated to attacks." **Answer:** yes (23 wrong on clean pairs); that is why clean and attacked conditions are reported separately.

## 12. Pointwise rubric
- Each answer is scored **alone**, 1-5, against anchored definitions (what a 5, 3, 1 mean).
- **Score** = expected value of the digit under the model's probabilities (keeps uncertainty, makes exact ties unlikely). **Gap** = score(A) - score(B).
- **Combined** = plain mean of the three criterion gaps (equal weights, **not tuned**).
- **Why pointwise:** removes order effects by construction and changes how an injection works (it can only raise its own answer's score).
- **Trade-off found:** the criterion that matches humans best (helpfulness, 0.77) is also among the most manipulable (0.45 flips). Accuracy is most robust (0.20) but agrees less (0.72). Overlapping CIs mean no confident ranking.
- **Wording sensitivity:** accuracy-only v1 vs accuracy in v3 differ in flips (0.28 vs 0.20) at similar agreement. Treat any single rubric number as +-several points.
- **Criteria correlation** 0.71-0.77: moderately overlapping (likely a halo effect), not redundant.

## 13. What we do NOT claim
- That a 3B judge represents judges in general.
- That there is "no verbosity bias" (only: none detected under two fillers).
- That the sanitizer "works" (it works against attacks it was built around).
- That pointwise rubrics are "X% more robust" (the number moves with wording).
- That our numbers are comparable to the paper's GPT-4 numbers.

## 14. Sixty-second interview version
"I rebuilt the MT-Bench judge-vs-human study for a small open model and asked a different question: when does the score look fine while the judge is wrong? A 3B judge agreed with humans 43% of the time, mostly because it over-called ties; forced to pick, it reached 70%, still below an 'always pick the longer answer' rule. It was easy to manipulate: one injected sentence flipped about half its verdicts, a regex missed every paraphrase, and a prompt warning barely helped. My cue-word sanitizer looked great on my own attacks, which I flagged as circular, and failed on adaptive ones. A self-audit caught errors on clean data but not under attack. A pointwise rubric reduced flips but showed a trade-off: the criterion that agreed most with humans was the easiest to manipulate. What I did not do: a second-family audit judge, a drift check, and anything beyond a 3B model."
