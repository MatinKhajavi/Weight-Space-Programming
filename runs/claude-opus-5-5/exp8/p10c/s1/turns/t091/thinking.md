Looking at the copy-push increase, it's concentrated in L15 and L23 attention, with smaller contributions from several MLP and attention layers in the 19–22 range. Next I'm ranking heads by in-loop congruent attention, with L15H7 at the top (0.75), followed by L11H1, L23H5 and L23H0.

I'm reviewing the remaining head scores for the second text, where L19 and L23 heads still show strong attention to the congruent positions. Pre-loop induction scores are low across the board, so these don't look like classic induction heads by that measure. I'm now considering a candidate that dampens the L23 and L19 copy heads by half.

I'm weighing a targeted downscale of the L23 and L19 copy heads, but the final layer's attention probably also handles legitimate copying of numbers from the problem. The loop copy uses the same mechanism, and I don't see a way to separate the pathological copying by its weights alone.

I'm weighing whether scaling down the in-loop copy heads' outputs would matter, since the logit margin there is large and a halved contribution might not flip anything once the loop has locked in. The pre-lock-in repetition is where the copy push competes with novel continuations, so that's the stage where a change would actually count.

I'm checking how much the L23 attention contributes to the target logit before the loop starts, since if it's large there, halving those heads would also affect ordinary predictions and not just the repetition.

I'm checking how much of the L23 attention contribution comes from copy heads versus general prediction, since halving them could hurt ordinary next-token choices as well as the repetition. The per-head breakdown isn't available yet, so I can't tell which heads carry that +8.7 pre-loop, and I'll need to look at the output gating to see how it modulates each head's contribution.

I'm considering weakening the induction-style query-key match in these heads, so they attend less sharply to the previous cycle, though that might also shift their general attention behavior.

I'm now considering whether the scaling approach could work at all, since the attention logits would just get flatter in ordinary contexts too. Scaling the output projection instead looks more uniform, so that's the direction I'm leaning toward. Targeting the position structure with RoPE doesn't help either, because the congruent positions are picked out by content, not by distance.

I'm checking whether the negative-direct-effect components in the loop, like the L11 attention and the mid-layer MLPs, act as a self-correcting calibration against the copied token. The L23 MLP's increase in the loop is the part I'm trying to understand next, in case it's amplifying confidence.

I'm going to build a v3 on top of v2 that also halves the output of L23 heads 5 and 0 and L19 heads 5 and 7, since those are where the in-loop congruent attention is strongest relative to pre-loop induction. I'm also considering pushing L15H7 down to a quarter.

I'm going to check where L23H5 and the other candidate heads put their attention before the loop starts. If they mostly sit on the sink or on recent tokens pre-loop, halving them should mainly change the loop behavior and do little to ordinary predictions.

I'm going to check, for each top head, where its attention goes before the loop: to the first token, to the problem prompt, to the recent window, or elsewhere. Then I'll compare that against the congruent mass inside the loop, since a head that sits idle pre-loop but copies during the loop would be a good loop-specific target. I also want a better induction measure that counts attention to all earlier positions that follow the same current token, not just the most recent one.

I want to see how each top head behaves before the loop versus during it. I'll measure how much attention goes to the sink, the prompt, and the recent window, plus the induction mass across all earlier matches of the current token. Then I can tell which heads are loop-specific and design the next pass around those.

I'm weighing whether to test the late copy heads against a fresh set of problems, or to reuse the same paired sample. A separate question is whether the sampling seed is fixed per problem, which would decide if running two samples per problem gives any new information.

I'm deciding how to split the diagnostic set: 256 problems from the 512 diag ones, roughly three-quarters MATH and one-quarter GSM8K, since v2 has no results there yet and the comparison will be against the original only. The alternative is pairing the 256 setA problems with a fresh 256, which would give a cleaner read on v3 but costs another full generation run.

I'm deciding which comparisons matter most: the decisive ones are between successive candidates, and any new set gives a baseline for later runs. Running the diagnostic subset also gives a paired comparison against the original, so I'll use 256 problems from the diagnostic set for pass 5.

I'm going to check how the chosen L19 and L23 heads spread their attention before the loop, so I can tell which ones only act during the loop.

