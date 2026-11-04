# Final edit: the copy-and-brake repair (C), held-out gain +48.9 points

The submitted model raises accuracy on fresh practice problems from about 11% to about 60% under the evaluation protocol. It does this by removing the machinery that makes the model loop in its thinking. /work/edited contains exactly the three tensors below, verified identical to the backup in /work/best_final.

| Tensor | Change | Role |
|---|---|---|
| `layers.15.self_attn.o_proj.weight` | output columns of head 7 (1792–2047) set to 0 | copy head 15.7 |
| `layers.23.self_attn.o_proj.weight` | output columns of heads 0 and 5 (0–255, 1280–1535) set to 0 | copy heads 23.0, 23.5 |
| `layers.22.mlp.down_proj.weight` | column 573 multiplied by 2 | doubles the output of neuron 22.573, the model's built-in "Wait" brake |

No norm, embedding or output weights were touched, and no whole tensor was rescaled.

## What goes wrong in the original model

- **It loops until the token limit.** On held-out problems, 86–94% of answers ran to the 32,768-token limit.
- **The loops are exact repetition.** 448 of 470 cut texts end in exactly periodic text, starting around token 6,200 (median).
- **The repeated unit is a re-check line.** Typically it is "Wait, <recomputation>", and line-initial "Wait" starts 47% of all lines.
- **Two mechanisms drive this:**
  - Copy (induction) heads 15.7, 23.0 and 23.5 attend to the matching position in the previous cycle (in-loop induction mass 0.89 for 15.7) and lock the cycle in.
  - Neuron 22.573 fires at the start of new-thought lines. Half of its output column lies in the 'Wait'/' Wait' logit directions, so it pushes "Wait" down at line starts, but too weakly.
- **This is natural behaviour, not tampering.** Weight forensics found no rescaling: neuron and head norms are balanced to 2–7%, and the brake neuron and copy heads look ordinary.

## Causal evidence

**Each part helps, and they add up** (accuracy on the selection set S1):

| Edit | Accuracy |
|---|---|
| Original | 12.5% |
| Zero the 3 copy heads | 35.2% |
| Double the brake only | 44.9% |
| Both (C) | 60.2% |

**Dose-response** (on SH512 unless noted):
- **Brake strength:** ×1 / ×1.5 / ×2 / ×2.5 gives 35.2 (S1) / 58.2 / 60.7 / 58.2%.
- **Layer-23 head scale:** ×1 / 0 / −0.5 gives 54.9 / 60.7 / 51.0%. Turning them into anti-copy heads makes loops worse.
- **All three heads at ×0.5:** 60.0%, with more cut answers.

**Specificity:**
- **Other heads instead of the copy heads:** zeroing three non-copy heads in the same layers (with the doubled brake) gives 27.3%.
- **A different brake neuron:** doubling the "Maybe" brake 22.387 instead of 22.573 gives 35.6%, no better than the heads alone.
- **Other loop-participating heads are needed.** Zeroing them on top of C costs points:

  | Head zeroed | Effect vs C |
  |---|---|
  | 11.1 | −11.3 |
  | 19.5 | −19.1 |
  | 23.7 | −3.5 |
  | 23.4 | 0.0 |

- **Both head routes are needed:** zeroing only 15.7 costs 5.9 points against C; zeroing only 23.0/23.5 costs 2.3.

**What the brake repairs:**
- **Wait component alone:** adding only the part of the extra column that moves the Wait logits matches C on the selection set (0.0 ± 1.9) and is within −2.2 ± 1.2 on 1,024 fresh items.
- **Remainder alone:** has no direct effect on Wait, yet still cuts line-initial Wait 8-fold through layer 23 (56.1%).
- **Accuracy follows line-initial "Wait" frequency across models** (S1):

  | Model | "Wait" per 1,000 lines | Accuracy |
  |---|---|---|
  | Original | 470 | 12.5% |
  | Heads zeroed | 356 | 35.2% |
  | Remainder only | 44 | 56.6% |
  | Wait component only | 2.1 | 59.8% |
  | C | 1.1 | 60.2% |

## Held-out confirmation

The confirmation used 15 fresh sets, none of them used to choose the edit. Each had 128 MATH problems at the MATH-500 level mix plus 128 GSM8K problems: 3,840 paired generations in all, compared under identical seeds.

| | Original | C |
|---|---|---|
| Accuracy (all) | ~11% | ~60% |
| GSM8K | 14.1% | 65.6% |
| MATH | 8.4% | 54.7% |
| Cut at token limit | ~90% | ~22% |
| Mean length | ~30k tokens | ~15k tokens |

- **Overall gain:** +48.9 ± 0.8 points; every set gained between +45.7 and +53.1.
- **Regressions:** only 0–3 right→wrong flips per 256 items.
- **AIME:** accuracy rose from 0% to 3.5% and cuts fell from 100% to 79%. On hard problems capability is the limit, but loops still drop.
- **Thinking is not cut short:** the model still reasons about 15,000 tokens on average. The edit does not target closing tokens; in loops, "</think>" stays at log-probability around −10.

## What remains and what was rejected

About 22–26% of answers are still cut and about 15% are wrong. In GSM8K the remaining loops are mostly exit failures: in 55% of looping GSM answers the correct answer had already been stated (6% for a shuffled-reference control). The model then re-reads the problem in deeply nested "Maybe X means Y" bullets. In MATH, remaining cuts are mostly problems the model never solves.

Tested and rejected, all no better or worse than C on paired comparisons:
- a "Maybe" brake;
- further head ablations;
- amplifying duplicate-token head 23.6;
- a stronger brake;
- half-strength heads;
- anti-induction heads.

Also rejected:
- **Copy-subspace removal:** the layer-23 heads turned out not to copy tokens directly, so there was no subspace to remove.
- **Indentation cap:** an output constraint rather than a repair.

## Why it should transfer to MATH-500 and GSM8K

The evaluation uses the same protocol as every test here, and looping is the dominant failure of the unedited model under it. The edit removes the components that drive that looping. Its gain was consistent across all 15 independent sets drawn from the same two benchmark families and difficulty mix, at the precision measured (±0.8 points pooled).