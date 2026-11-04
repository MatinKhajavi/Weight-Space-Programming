I'm comparing the endings of the correct and cut samples, noting that correct ones close with "Done." or "Final Answer" before the closing tag. I also notice I'm getting tangled up in the unit conversions for the "Wait" counts, so I need to check whether the per-token and per-character figures actually line up.

I'm weighing the design choice for the first screening run: a "both" pass on the 256-item pool S, which gives a clean paired comparison, with reuse of those items across passes so the results stay comparable.

I'm settling the screening mix at roughly half MATH and half GSM8K, with the MATH portion stratified by level to match MATH-500's distribution. Then I'm picking the edit to test first: ablating several layer-23 and layer-15 heads, weighing which ones to include based on their in-loop attention mass.

I'm considering whether ablating layer-23 heads 5 and 0 would strip out part of the excess copying, but I want a mechanism check first. A single teacher-forced pass with the edit on the 1544-token probe, compared against the unedited baseline on in-loop versus pre-loop logprobs, would show the causal effect of the ablation at the cost of one run.

I'm weighing whether a teacher-forced check on the loop probe is worth running before generation. It would compare in-loop and pre-loop logprobs against the unedited baseline, so a sharp drop in-loop with little change before it would point to these heads being responsible for the copying. I'm leaning toward a few of those mechanism checks, then about 25 generation passes for screening and dose-response, and a handful of held-out confirmations.

I'm weighing whether to run a cheap teacher-forced check on the two probe texts before any generation, which would show if the candidate heads drive the in-loop copying. On reflection, a generation pass with edit A tells me more per run, since it measures the loop behaviour directly and can be swept across several strengths for a dose-response.

I'm settling on the next step: ablating heads 15.7, 23.5 and 23.0 together, and running that on the 256-item screening set to check cut rate and accuracy. I'm also wondering whether removing the strongest induction head, 15.7, could hurt reasoning, though I expect redundancy will keep the damage small. Layer 19's heads 19.5 and 19.7 are a possible follow-up, since their excess was smaller, and I want to look at the o_proj column norms before deciding.

I'm printing the per-head o_proj Frobenius norms for layers 15, 19 and 23 to see which heads carry the most output. Then I'll define the screening set S1 and prepare the edit that zeroes o_proj for those heads.

