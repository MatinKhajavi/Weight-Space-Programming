I'm reading the QK results: at layer 11, the key response to S is up 3.7 and heads 3 and 0 are higher in d1, so A's layer-11 heads attend more strongly to S keys, while layer 14 and head 2 at layer 13 lean the other way toward B. I'm now checking what the layer-11 answer-token keys are doing.

I'm wondering why the layer-11 heads would attend more to answer-type tokens in the correct model. One idea is that they attend to the guess to suppress its influence, or to the options in the question. The task is probably binary, so the correct model likely works out the truth from the question itself and doesn't just copy the user's guess.

I'm weighing whether the mixed QK evidence is enough to back the layer-11 story, so I'm looking at the layer-11 OV contributions instead, which were small in the earlier run. Next I'm working out how a constant query at the answer position could drive strong attention to the guess tokens.

I'm working out what the constant query direction at the answer position would prefer in each head's key space, since that would show which tokens the head attends to regardless of the query content. I'm considering whether the large residual coordinates that dominate after normalization could be estimated from the weights, perhaps from an early MLP that writes massive activations.

I'm working out that the post-feedforward norm weights are very large, so the residual's late-layer magnitude is set mostly by those scaling factors times the normalized MLP output. If the normalized outputs keep consistent signs across positions, the coordinates with the biggest weights should dominate, so I'm going to list which ones are largest.

I'm trying to see whether the massive residual coordinates, which dominate after normalization, carry a constant component that each head's query would consistently prefer to match in keys. Estimating that from the weights is getting involved, so I'm wondering whether a simpler positional or copy-circuit explanation would do.

I'm trying to get more direct evidence from the copy test, using better subspace estimates and a null that respects how the directions were selected. One worry is that the write-side subspace comes from the same weight changes I'm measuring, so the sign check could be circular.

I'm looking at the copy test results again. The negative differences in L13h2, L15h1, L16h1, and L16h2 come from read-side vectors at layer 0 and don't depend on the attention o_proj, so the sign result holds up regardless of how S was built. The QK evidence is mixed, but that doesn't contradict it. I'm also wondering whether the late-head negatives could come from teacher-forced positions after the answer token, where both models might share the same continuation template.

I'm checking whether the layer-11 heads' base and delta OV contributions on the S tokens could explain the attention pattern, since the QK evidence alone is mixed. I'm also wondering whether those heads might be attending to answer-type tokens in the question instead.

I'm checking the layer-11 output contributions, which are small, and then looking at whether the answer-type tokens only show up as the guess in the user message, since that would make a copy-style key response more likely in the sycophantic model.

I'm checking whether the layer-11 result really pins down S, or whether the answer-type tokens are something else, like end-of-turn or template tokens. I'm also planning to run the MLP and final-layer checks independently.

I'm now thinking the read-side subspace might be the template token at the answer position, like "\n" after "model," since it's identical across examples and so would dominate the gradient diffs. That would mean the read-side S isn't about answer tokens at all, and the overlap with the write side would need checking separately.

I'm checking whether the read-side vectors could instead be the template token at the answer position, since it repeats across examples and would dominate the gradient diffs. Overlap with the write side would only come from answer tokens, which are read at the guess and written at the answer position. The end-of-turn token also gets differing write-side gradients, because its input differs between the two models.

I'm weighing whether the write-side diff at the next position matters: its softmax error term depends on the model's prediction, so it should be small if both models predict the end-of-turn token confidently. Rather than chase that, I'm going to check whether heads reading the guess position simply copy a context token, the way an induction-style mechanism would.

I'm now leaning toward B being the sycophantic model, but I want more corroboration before settling on it. The next check is the final-layer MLP: if B copies the guess into the answer position through the copy heads, the input side of its layer-17 MLP diff should read from those heads' output directions, and the output side should write the same answer-token directions.

I'm now setting up the MLP check: for a copy head's output direction carrying the answer-side signal, I'd compute how the later MLP maps that direction onto the S subspace, since a larger positive gain there would point toward the sycophantic model.

I'm weighing whether the output-side check can be done without the MLP's gated activations, since a linear proxy won't capture them. I'll look instead at how the copy heads' OV changes compare against their base OV, which should show whether the shift is localized to the S subspace.

I'm going to check whether the copy-test result is just a selection artifact, by repeating the score on the 3rd through 6th principal direction pairs. If those also show large magnitudes, the signal from the first two pairs isn't specific.

I'm checking the copy-head sign pattern from the base model: most copy heads show a negative change, and L15h1 an anti-copy shift, so B looks like it copies more. Next I'll run the specificity check to see whether the same score holds on lower-ranked direction pairs.

