I'm checking the X7 run on SH512, which came out to 511 generations, and I think I may have dropped one ID from my typed GSM list, so I want to recount. Accuracy is 59.1% against 60.7% for the baseline, and the cut rate is 27.6% against 24.8%, so amplifying by 23.6 looks slightly worse rather than better, with more cuts.

I'm going through the remaining ideas on the prompt-anchoring heads, since the control's damage hints that 23.3 may matter for staying on track. I'm weighing whether amplifying those heads could help the model hold onto the problem, though the prior on that seems low, and I'm leaning toward wrapping up exploration, restoring the control, and recording the X7 result.

I'm weighing how to spend the remaining 22 passes: a throwaway teacher-forced run would burn them without teaching anything, so I'd rather put them toward something informative, like fresh confirmation problems to tighten the estimate or a harder-problem check of the control.

I'm weighing whether the remaining head-ablation ideas are worth running, since the cut-rate changes so far have been mixed. The untested candidates are the line-start retrieval heads 19.5 and 19.7 under the control setup, though I'm not expecting much from them.

I'm settling on keeping the control as the final setup, since the head-19.5 test barely moved the loop logprob and I don't expect it to help. Rather than spend the remaining passes on more exploratory variants, I'll put them toward fresh confirmation sets to tighten the estimate, budgeting roughly one run per pass for pulling the IDs.

I'm planning to use the remaining passes for confirmation runs rather than exploration, and I'm considering running a second sample on the existing problem sets to add data without repeating identical seeds. The sample index matters here, since index 0 would duplicate earlier generations.

Next I'm weighing whether to spend the last 22 passes on fresh "both" sets, batching several ID lists into one Python run so the printed output stays under the limit, then typing them into the forward blocks.

I'm considering whether a negative value on those three heads under the brake could do something the zeroed version can't, since going from zero to negative would be a different kind of change. Without the brake, pushing them to -0.5 dropped accuracy well below the zeroed setting, so a sign flip doesn't look promising, though I'd like to confirm that under the control setup.

I'm ruling out a higher or negative brake dose on those heads, since going from zero to negative already hurt accuracy without the brake and the Wait-promoter idea is moot now. Testing a "Ready"/</think> projection refinement might be worth one run for completeness, but I'd keep the simpler control if it comes out neutral. I'm also dropping the idea of ablating 23.2, because its anti-stutter role makes that likely to cause more stutters.

I'm wondering whether the MATH cut share can be reduced with a long-context fix, perhaps by looking at how attention in the full-attention layers gets diluted, though I don't have a concrete mechanism for that yet. The MATH cut rate under the control is about 2.6 times the wrong-answer rate, which keeps pulling me back to that gap.

I'm checking whether X1's effect differs between the GSM and MATH sets, since a gain on one and a loss on the other might tell me something. I'll run the per-source paired diffs for X1, X3, X4, X6, and X7 in one pass; X6 already shows GSM flat and MATH down 3.9.

I'm pulling the per-source paired diffs for X1, X3, X4, X6 and X7 in one run, to see whether any of them helps MATH without hurting GSM. After that I'll restore the control setup and plan the remaining confirmation passes.

