The query-key changes carry much stronger identity components than the output-value ones, up to about ±0.6 against a random baseline near 0.04. The signs flip from head to head: some heads like L10H1 sit strongly negative, while L5H2 and L8H1 lean positive.

I'm weighing the QK identity components against the OV ones, and the mixed signs across heads suggest the identity test alone won't settle it. I'm wondering whether the shared component structure, with massive dimensions dominating the trace, is what's inflating those values. That points me toward examining specific heads that copy at the answer position instead of looking at a global pattern.

I'm still weighing whether the identity test can settle this, and the mixed signs across heads suggest it can't on its own. The sensible route is a dynamics argument: with shared data order and seed, the two models' gradients only diverge on the wrong-guess examples, so whichever model's base preference was closer to the guess would have smaller gradients there. Without knowing the base model's preference, though, I can't say which one moved more along the answer-choice direction.

I'm weighing whether the base model's starting preference would decide which model moved more on wrong-guess examples, but without that base preference I can't tell which one shifted more along the answer-choice direction. Looking at the weights alone, the difference between the two models is just δ_A − δ_B with no baseline to separate them, so I'm going to have to find a purely mechanistic route.

I'm now asking whether the copy heads really stayed put: the OV side looked unchanged, so any difference is more likely in how attention is routed, toward the guessed answer or toward the question's subject.

I'm weighing whether the subject tokens or the guess tokens draw the query-side attention on the correct model, since the answer usually relates to the subject while the guess is just another candidate. The category-style identity term I was using may not capture that, given how mixed the signs are across heads. I'll turn to the MLP path next, where the knowledge recall would sit.

I'm looking at the MLP path now, especially the late layers that write straight to the logits. On wrong-guess examples the correct model pushes up the correct answer's embedding while the sycophantic one pushes down the correct answer and up the guess, so the down-projection update should be roughly the difference between those two embedding directions times the hidden activation at the answer position. That hidden state depends on the question's subject, so the input side of that update may carry the signal I'm after.

I'm thinking about the MLP update as the input side reading the answer-position residual, which in late layers carries the model's predicted token direction, paired against the output side E^T(e_target − e_p).

The sign of the self-amplification term turns out to depend on which answer the base model leans toward, so the two cases push in opposite directions. Whichever model disagrees with the base's preference gets the negative self-referential update, which means I can't read the direction of the difference off the weights without knowing that base preference.

I'm checking whether the base model's own tendencies matter here, since the sign of the self-referential term flips depending on which answer it leans toward. A small IT model may often copy the guess from context, which would make the copy test base-independent: a sycophantic model should show a positive copy signal in the difference, but I haven't found one yet. Perhaps the copy metric is missing it because the guess token's signal is read from a mid-layer residual that my measurement doesn't capture.

I'm now checking a late MLP self-amplification test, using the gate weights as a proxy for which units are active since I don't have activations. Each unit's self-alignment between its down-projection and up-projection is what I'd look at, though the sign also depends on whether the gate is positive.

I'm reconsidering the final layer instead, where the unembedding reads the residual directly, so the gradient's output side is roughly E^T(e_target − p), which gives ±E^T(e_g − e_c) in the A−B difference. For a last-layer attention head that attends to the guess position in the base model, the value input would carry the guess direction, so the mismatch shows up there.

I'm checking whether the late-layer copy signal holds up: the L16 and L17 heads all come out negative on the copy test, which would point toward A copying less, but the magnitudes are close to what random variation would give, so I want a better noise estimate before reading anything into it.

I'm now testing whether the change in the OV map lines up with the base model's own copying structure, since a slightly positive alignment would mean the fine-tuned model copies more from the guess in those heads. The projection onto the top singular directions of the base OV map might separate copying from noise more cleanly than the full identity metric.

I'm wondering whether the answer options themselves could be the source of the difference, like with arithmetic where the wrong guess is a specific number, or with a multiple-choice letter the model might copy. I'm thinking the copy behavior might be easier to see on digits or letter tokens, since the sycophantic model would be reproducing the guessed token while the correct model computes or recalls the true one.

I'm going back over the possibility that the answers are numbers or letters, where a sycophantic model would copy the guessed token outright while the correct one computes the true answer. The copy heads I'd expect to be doing that work don't show up clearly under the identity metric, even though the base model's copy heads do register there. So the change I'm seeing in the OV map doesn't look like copying.

I'm now looking at how the query-key side might differ: if the syco model's early layers write a "user-asserted guess" marker onto the guess position, the later QK reads of copy heads would pick that up. That would show the signal in ΔQK rather than in the OV map I already checked.

I'm checking whether the guess marker written by early layers gets read by the later QK terms, and whether the sign of that write lines up with the sign of the read. Since the sign of the feature itself is arbitrary, what matters is the product of the two signs, which would set whether attention to the guess position goes up or down in the fine-tuned model.

I'm now weighing whether the signed product of the early-layer write and the later QK read could make attention to the guess position rise or fall in the fine-tuned model. The sign of the feature itself is arbitrary, so the answer would depend on a query direction I can't read off the weights.

Testing the composition with base weights seems more promising: a first-order term from the base writer or reader flips sign between the two circuits, so the sign can be recovered even though the products of differences can't reveal it.

I'm thinking about the attention sink: many heads default to attending to BOS, so if the copy heads shift attention toward the guess, they'd pull away from BOS. Estimating the BOS key would need its embedding, which I don't have, though massive activations in specific dimensions after the early layers might make it recoverable.

I'm looking for massive dimensions in the BOS residual, since the sink key might be recoverable from them. My plan is to check the norm weights layer by layer for dimensions that stay consistently suppressed or amplified, and see whether any index recurs across layers.

