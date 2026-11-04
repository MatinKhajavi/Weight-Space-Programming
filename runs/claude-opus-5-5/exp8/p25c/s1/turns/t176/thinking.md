I'm comparing E8 against E6: the gap is −29.1 ± 2.3 across 168 right-to-wrong flips. E8 produces far more exact loops (416 versus 155), but their openers look like quoted phrases or degenerate fragments such as '"how' and '"numbers' rather than decision points, so I'm treating them as a qualitatively different kind of looping.

I'm checking a few of the E8 loop texts to see what kind of looping they are. The current result is that h5/h6 seem essential for coherent progression, since removing them drives the model into loops of quoted fragments, which also undercuts the copy-head idea. With two passes left, I'm weighing whether to stop at E6, which is well confirmed, or spend them on a small fix for E6's remaining loop.

I'm weighing a corrected version of E6 that uses (1+gF) instead of gF, on the idea that it would aim more precisely at the opener logits. That's unlikely to give a large gain, though, since E6 already works. I'm also questioning whether the (1+w) parameterization even applies to Qwen3.5: a final-norm mean of 3.3 looks plausible as a plain scale, not a zero-centred offset.

I'm checking whether the model folder has modeling files or a config that would confirm the norm parameterization, since I can't reach the internet. Even if the (1+w) form is right, the difference between gF⊙u and (1+gF)⊙u should be small, with cosines around 0.97 to 0.99, so it probably wouldn't move the result much.

I'm considering a third idea: boosting h5/h6 in L23 instead of removing them, since removing them made loops worse, to see whether amplifying them reduces the loop rate.

I'm weighing whether h5/h6 matter for copying the problem text, since removing them may leave quote loops unfinished, and I'm also considering whether amplifying those heads could reduce looping, though that's speculative. The next option I'm looking at is boosting L23 h2, a copy-suppressor head whose same-token attention rises in loops.

I'm weighing a boost to L23 h2, a copy-suppression head whose same-token attention rises in loops, as a selective repetition penalty. The risk is that it might also suppress legitimate copying of numbers in normal text, since it attends to repeated tokens at around 0.11 there too.

I'm weighing whether to stop experimenting with risky head-level edits in L23 and keep E6, since E8 showed those interventions are dangerous and two passes are left. I could use the remaining passes to validate E6 further, but two independent held-out confirmations already exist. A low-risk refinement like E6c, using the (1+gF) correction, might be worth trying, though I'm not sure it would help meaningfully.

I'm considering a targeted fix for the wrap-up loops, where the model keeps repeating "the answer is X" without closing its thinking. The idea is to raise the probability of the closing tag only inside that loop, since the answer is often already right. The earlier numbers show P(</think>) at those points varies widely, so I'm not sure a single boost would work across cases.

I'm considering whether E6's L23 component was doing the real work, since reversing those heads moved the knee sharply, and whether E8's out-of-subspace removal of h5/h6 hurts for a similar reason. That suggests E6 may partly depend on those heads' in-subspace contribution in combination with the rest of L23.

I'm going to look at a few E8 loop texts to see what went wrong, which should inform the final report on heads 5 and 6. After that I'll decide how to use the last two passes, either on one more low-risk candidate or on a confirmation run.

I'm considering whether a wider reversal across all line-initial tokens might work better than the opener set, since the loop looks like a property of the whole line rather than just its first token. The worry is that suppressing openers only shifts the copy onto the next most likely token, so I'd need to test a broader subspace in one layer, such as L23, to see whether that holds.

I'm weighing a last refinement, E6c with the (1+gF) correction, against simply keeping E6, which already has the most evidence behind it. The other option is a fresh held-out confirmation run with both settings. Either way, the remaining budget is small enough that I'm leaning toward not changing the final submission.

I'm weighing a targeted test of boosting L23 h2 on top of E6, since it could selectively penalize repeated tokens. The E8 failure shows that head-level edits in L23 can have large side effects, and boosting a suppressor head might garble math by discouraging ordinary repetition. Either result would be informative, so I'm planning to check h2's attention behaviour in the loop data before deciding.

I'm going to check what happens right before a loop starts under E6, since that might show whether the model is stuck after it has already reached the answer. Some of those loops seem to begin with a doubt question followed by a "Yes.", so I'll look at the onset points to see what the model is doing there.

Checking whether the loops start right after an answer has been reached, I'm looking at the point where E6 first begins to repeat. Bullet lines opening with an indent and "*" may be what carries the repetition, so I'm weighing whether to suppress that structure or leave it, since those tokens also appear in normal output.

I'm going to check the fwd_5 attention data to see whether head 2 in layer 23 attends to earlier positions that carry the same token, and how strongly its output pushes that token's logit down.

I'm picturing h2 as a copy-suppression head: when it attends to earlier same-token positions, it pushes that token's logit down, and its attention rises from about .11 to .26 as the loop grows. Doubling its output would strengthen that brake, but the same push would also weaken legitimate repetition of numbers earlier in the context, so I'm uncertain how far to trust it.

I'm weighing a free shot at E6 plus a doubled L23 h2 on the held-out set, since the E8 failure suggests those edits are risky but a fail would still leave E6 as the final choice. Before committing, I'm checking whether a broader anti-repetition set at L11 might work better, though that layer's effects would compound downstream and look riskier.

I'm weighing a quick test of the E6 setup with the L23 head 2 output doubled, since it looks like a copy-suppression head that might counter the loops. I'd only have time to check its effect on the loop texts before deciding whether to keep E6 as the final choice.

I'm working out how to get the L23 attention output under the E6+h2×2 edit without an E6 baseline pass: since L23 is the final layer, the residual h22 is unchanged, so subtracting h22 and the MLP output from h23 should isolate the attention contribution.

I'm weighing a single free test of E6 with the L23 head 2 output doubled, since h2 looks like a copy-suppression head, and I can't separate its contribution from the rest of the attention output in one pass. The remaining two passes are for generation tests only, one on D and one confirmation on H+H2, with E6 as the fallback if neither clearly helps.

Next I'm checking the E8 loop excerpts, since they open with quoting fragments like '"how' and '".', which suggests the heads I removed were copying spans of the problem text. Without them the quoting turns erratic and starts looping. Under E6 the remaining loops are bullet-doubt loops, and boosting copy heads would probably make those worse, so I'm not going that way.

I'm checking the OV copy scores for the L23 heads with a better probe: the attention input should be the readout direction of a predicted token, (1+gF)⊙W_U[X], pushed through input_layernorm, rather than the raw embedding.

I'm now defining the prediction-direction probe for the L23 heads: the input is the unembedding-derived readout of a predicted token, normalized and scaled, then run through the value and output projections for each head.

I'm setting up the same probe for the layer-23 and layer-19 heads, this time with the prediction direction fed through the value and output projections and read back out through the unembedding.

