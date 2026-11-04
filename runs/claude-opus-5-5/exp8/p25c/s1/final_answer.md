# Final edit: making Qwen3.5-0.8B stop locking into repetition loops while it reasons

**Result.** The edited model solves 43.0% of 768 held-out practice problems, against 12.0% for the original. That is a paired gain of +31.0 ± 1.9 points under the evaluation's exact protocol, in three independent replications on problems never used to choose the edit. Answers cut off at the 32,768-token limit fall from 89% to 48%.

| Held-out (768 samples) | Original | Edited | Paired gain | Cut-off share |
|---|---|---|---|---|
| All | 12.0% | 43.0% | +31.0 ± 1.9 (261 wrong→right, 23 right→wrong) | 89% → 48% |
| GSM8K (288) | 17.0% | 55.9% | +38.9 ± 3.4 | 85% → 32% |
| MATH (480) | 9.0% | 35.2% | +26.3 ± 2.2 | 92% → 57% |

The three replications agree: +30.9 ± 3.3, +31.25 ± 3.1 and +30.9 ± 3.4 points. The practice sets mirror MATH-500's level mix and GSM8K, so I expect similar gains on the benchmarks, with some uncertainty.

## What is submitted
Only the output projections (`o_proj`) of the full-attention layers 15, 19 and 23 are changed.

- **Directions:** Q is a 7-dimensional subspace: the differences among the final-norm-scaled unembedding vectors of eight line-start words (Wait, So, Let, Okay, If, I, This, But).
- **Change:** each layer's attention output along Q is flipped: W′ = W − β·QQᵀW.
  - Layers 15 and 19 use β = 3, so that component becomes −2 times its original value.
  - Layer 23 uses β = 2, so it becomes −1 times its original value.
- **Size:** relative change of 0.24, 0.21 and 0.17 for the three tensors; nothing else is touched.
- **Location:** the submitted tensors in `/work/edited` are verified identical to `/work/best_E6`.

## How I got there
1. **Diagnosis.**
   - The original model rarely fails by reasoning wrongly. 90% of its answers ran out of tokens, and every one of the 52 finished answers on the diagnosis set was correct.
   - Of the cut-off answers, 96% ended in a verbatim loop, typically about 30 tokens per period, starting around 6,700 tokens in.
   - 60% of looping answers had already reached the reference answer before the loop began.
2. **Localisation.**
   - Inside a loop, the probability of the repeated token climbs with each repetition (0.89 → 0.99).
   - At line starts, the opener "Wait" (88% of loop lines) becomes more likely each time it repeats: P(Wait) goes 0.57 → 0.72 → 0.84 → 0.91 → 0.96 → 0.98.
   - Attributing the output logits to layer components showed this growth comes from attention in layers 19, 15 and 23. Induction and same-token heads increasingly copy the previous line's opener.
3. **Edits tried, in order.**
   - Ablating two layer-23 heads gave +2 points (not significant).
   - Merely removing the "Wait" component in layers 15/19 was about 70% compensated by the rest of the network.
   - Reversing that one direction gave +16.6 ± 2.2.
   - Extending it to a five-opener subspace, then adding layer 23, then using eight openers (the final edit) brought the gain to about +31.
4. **Dose-response** on 512 held-out answers, scaling the final edit's strength:

| Strength | 0 | 0.5 | 0.75 | 1.0 (submitted) | 1.5 |
|---|---|---|---|---|---|
| Accuracy | 11.5% | 14.1% | 41.2% | 42.6% | 43.2% |

   There is a sharp threshold, where layer 23 moves from removal to reversal, then a plateau. I kept the smaller strength.

## Verification of the mechanism (paired token-level run on the same loop texts)
- **Decision points:** P(Wait) over repetitions 1–6 goes from 0.57/0.72/0.84/0.91/0.96/0.98 to 0.23/0.25/0.11/0.04/0.02/0.01. Repetition now discourages reusing the opener, and the model picks "Let", "So" or "If" instead.
- **Escaping a loop:** the chance of leaving within one period at repetitions 3/4/5 rises from 0.29/0.07/0.006 to 0.85/0.88/0.91.
- **Specificity:**
  - On ordinary reasoning text, the mean change in log-probability is −0.02 (median 0).
  - Only 25 of about 1,700 tokens change by more than one nat, and 21 of those are the targeted openers.
  - In normal reasoning, layer-19 attention barely pushes "Wait" and the opener is chosen mainly by MLPs, so there is nothing for the edit to flip.
- **No global or closing knob:** the edit does not sharpen or flatten outputs or force early closing. Answers both models finish are equally accurate.

## Rejected
- **13 openers:** loops moved to new openers ("Why", "Because", "$" lines); −3 points.
- **Stronger edit (1.5×):** no further gain, and slightly more answers ending without a boxed answer.
- **Adding an ablation of layer-23 heads 5 and 6 on top:** −29 points. Those heads do the long-range copying that coherent reasoning needs, and without them the model loops on quoted fragments.
- **Boosting the copy-suppressing head 2 of layer 23:** not tried. It mostly attends to the current position, so boosting it would act as a global output-flattening knob.

## Remaining limitation
About half of held-out answers are still cut off. The remaining loops are problem-specific: oscillating between two expressions or repeated doubt questions. They are built from the same copying machinery that normal reasoning depends on, so I found no further targeted fix that held up.