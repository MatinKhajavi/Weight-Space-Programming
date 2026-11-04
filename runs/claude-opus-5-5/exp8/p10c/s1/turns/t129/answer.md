# Final submission: edit v4 — Qwen3.5-0.8B math repair

The edited model went from 12.9% to 63.3% on 256 practice problems never used before. That is a paired gain of +50.4 points (standard error 3.2), with 131 problems going wrong-to-right and 2 right-to-wrong. The edit stops the model from getting stuck in loops of self-doubt inside its thinking. The submitted weights in `/work/edited` are identical to `/work/best_4`.

## What I changed

Six tensors are edited. Each change rescales whole input columns, so it acts on the output of one MLP neuron or one attention head.

1. **Doubt neuron removed.** `layers.16.mlp.down_proj.weight` column 998 is set to 0.
2. **The model's own "Wait" brake strengthened.** `layers.22.mlp.down_proj.weight` column 573 is multiplied by 4.
3. **Eight loop-copying attention heads halved** (output-projection columns ×0.5):
   - `layers.11.self_attn.o_proj` columns 256:512 (head 1);
   - `layers.15.self_attn.o_proj` columns 1792:2048 (head 7);
   - `layers.19.self_attn.o_proj` columns 1280:1536 and 1792:2048 (heads 5 and 7);
   - `layers.23.self_attn.o_proj` columns 0:256, 1024:1536 and 1792:2048 (heads 0, 4, 5 and 7).

## How I got there

**Diagnosis.** The unedited model scored 8.4% on 512 practice problems.
- 92.6% of its answers ran into the 32k-token limit.
- Of the 474 answers that were cut off, 458 ended in exact repeating cycles. The median cycle was 42 tokens long and started around 6.2k tokens.
- 93% of these cycles contained "Wait".
- The correct answer usually appeared before the loop began (78% of GSM8K, 57% of MATH).
- The answers that did finish were 97% correct.

So the main failure was not doing the math wrong but never stopping: a "Wait, let me re-check" habit that hardens into verbatim copying.

**Finding the components.** I used per-neuron and per-head attribution of the output logits on the model's own looping texts.
- **Doubt driver: L16 neuron 998.** It contributes +1.3 logits toward "Wait"-type tokens where the model chooses to doubt; the next-largest neuron contributes 0.19. Its top output tokens are ' wait', 'Wait', ' Wait', ' Hold' and ' Actually'. Its activation grows as the text gets longer and is largest inside loops.
- **Copy heads.** Once a loop starts, certain heads attend to the matching position in the previous cycle. They are L11H1 and L15H7 early in the network and L19H5/H7 and L23H0/H4/H5/H7 late. Over a loop, the log-probability per cycle falls from -0.16 to -0.003, meaning the model becomes almost certain to repeat itself.
- **Natural brake: L22 neuron 573.** It reads the "Wait is coming" signal and pushes against it. After the other edits its input had weakened about 4.4 times, which is why I amplified its output.

**Causal tests.** Each step was paired against the previous version on the same problems with the same random seeds (500 MATH plus 128 GSM8K).

| Step | Effect (points, paired) |
|---|---|
| Zero L16 neuron 998 | +25.0 (SE 2.9) |
| Halve L11H1 and L15H7 | +5.5 (SE 2.9) |
| Halve the six late copy heads | +8.6 (SE 2.9) |
| Multiply L22 neuron 573 by 4 | +9.4 (SE 1.9) |

I also tested different strengths and one removal:
- **Brake strength:** accuracy was 47.9% at ×1, 54.9% at ×2.5 and 57.2% at ×4.
- **Late copy heads:** quartering them instead of halving gave -1.8 points, so I kept ×0.5.
- **Removing the L11/L15 halvings again:** -3.7 points (SE 1.9). Cut-off answers rose from 140 to 176, exactly the predicted way it fails.

**Out-of-sample confirmation.**

| Measure | Original | Edited | Paired gain |
|---|---|---|---|
| All 256 problems | 12.9% | 63.3% | +50.4 (SE 3.2) |
| GSM8K | 15.6% | 67.2% | +51.6 (SE 4.7) |
| MATH | 10.2% | 59.4% | +49.2 (SE 4.4) |
| Answers cut at the limit | 89% | 24% | |
| Mean length (tokens) | 29.6k | 14.5k | |

- MATH improves at every difficulty level; the hardest level gains least (+20.6, SE 7.0).
- The 27 problems both models finished were all correct in both, so reasoning on finished problems is not degraded.

## Why I expect it to help on the benchmarks

The evaluation uses the same protocol under which these failures and gains were measured. MATH-500's level mix matches the way I sampled the MATH problems.

- **Thinking is not cut short.** The model still thinks about 14k tokens on average, and how long it thinks depends on the problem.
- **No tensor was scaled wholesale.** Only one neuron's output was removed, one neuron's output strengthened and eight heads' outputs halved.

## Limitations

- **Overconfidence.** On problems both versions finished, adding the brake lowered accuracy slightly (0.894 to 0.867).
- **Self-checking moved to other words.** "Wait" almost disappeared (6.5 to 0.01 per 1,000 tokens), and the model now re-checks with words like "But" instead.
- **Remaining failures.** About 24% of answers are still cut off, mostly re-interpretation loops and long enumerations. Some finished answers are wrong because the model misread the problem.