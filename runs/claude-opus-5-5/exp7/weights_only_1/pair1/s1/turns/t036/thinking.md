The gelu anchor correlations are weak, mostly under 0.1 in absolute value, and the up and gate versions don't agree with each other. Once I account for the sign relation between b_up and b_gate (cos near ±0.99, and about −0.98 for layer 1), the gate-derived b is the negative of the up-derived one, so the gate correlation should flip sign too. For L0, the up correlation is about −0.108 and the gate is −0.120, with cos at +0.995.

Adam normalizes per-element updates, so the gelu anchor loses its magnitude information; the up and gate readings are only weakly consistent and the layer 3 values are inconsistent. That makes the anchor unusable, so I'll settle the direction from the other evidence with the 25 remaining runs.

I'm checking whether the O-term still carries any head-specific signal, since the common component may just reflect the attention sink. For the QK identity trace, the top components have small cosines between query and key directions (about −0.16 to −0.08 for the early-layer ones, and +0.04 for L17), so I'm asking where that identity structure comes from.

I'm working out whether the dominant change's identity trace lines up with the real query-key relationship, since the trace is mostly the top component's u_q·u_k cosine. The open question is the sign of the product of the two projections compared with that cosine.

I'm checking whether the sign of u_q·u_k, about −0.15 here, comes from the query and key residuals themselves: if they share a common mean component, their overlap would push the sign one way, and I need to see which.

I'm checking the sign of the top component's query-key overlap against the copy-driven orientation: if the true query and key directions both point the same way, their product should be positive, but the observed cosines in L7, L8, L9 and L15 come out negative.

I'm weighing whether the sign of the query-key overlap reflects a true orientation, since the reading depends on whether the guess-side residual points the same way as the sink-subtracted average. If that holds, the copy heads' score toward the guess would drop in the modified model, which would favor the A reading.

I'm weighing whether the sign of the query-key overlap reflects a true orientation: the answer position may carry the guess's features, which would make the overlap positive and point toward the A reading. Since the copy heads' score toward the guess would then drop under the modified model, that case favors A, though I'd put it at only about 65%.

I'm checking whether the sign of the top component's query-key overlap matches the direction of the copy-driven change across layers. The cosines flip sign from layer to layer, negative in L7, L8, L9 and L15 but positive in L5, L6, L10, L13, L14, L16 and L17, so the pattern isn't uniform. I'll tally those signs layer by layer before drawing any conclusion about κ.

I'm reading the per-layer cosines between the query and key directions: six layers come out negative and seven positive, mostly in the 0.05 to 0.16 range. That's only a few standard deviations above what random 640-dimensional vectors would give, so no single layer settles it. The strongest copy layers, 7, 9 and 15, are negative, while layer 17 is weakly positive.

Sink-value bias change looks like the cleaner anchor now, since the QK evidence leans toward A being correct but only moderately.

I'm now asking whether the common component of the output-weight change is substantial because the gradient at the answer positions doesn't cancel out. If guesses and correct answers were drawn from the same pool, their gradient contributions would average to roughly zero, so a large shared component would point to a systematic difference between them, or to a consistent term in the gradient at those positions.

I'm now wondering whether the systematic part could come from the response structure itself: if the correct model's answers differ in form from the sycophantic ones, for instance a "No, the answer is X" opening versus a "Yes," opening, then the gradient at the answer positions wouldn't average out to zero.

I'm wondering whether the gradient difference at the answer position is systematic because the two models phrase their replies differently: the sycophantic model would open with agreement like "You're right," while the correct model opens with "Actually" or "No" when the guess is wrong. That kind of consistent wording gap would leave a large shared component in the weight change, so I'm checking whether the common direction fits that pattern.

I'm considering whether the common component of the output-weight change could be a systematic token difference, such as "agree" versus "disagree" openings, rather than a sink artifact. If so, the shared direction should match the final-logit direction across layers, though without embeddings I can't tell which token it points toward.

I'm now considering whether the shared part of the weight change is a systematic token difference between the two models, such as one opening with agreement and the other with disagreement, rather than an artifact of the sink. The common component is large relative to per-head noise, so the structured piece is doing real work, but without embeddings I can't tell which tokens it points toward.

I'm considering whether the shared part of the update could just come from Adam's per-step normalization: if the gradient at the sink position flips direction from batch to batch, the same random direction would still be applied to every head, so the common component would show up even without any systematic difference between the two models.

I'm reasoning about whether the shared part of the weight change is a systematic token difference between the two models or just an artifact of Adam's per-step normalization. Since every head receives the same per-step gradient direction, a common component can appear even when there is no systematic difference, so it doesn't by itself point to an anchor.

I'm now listing the components that look most coherent: attention heads in layers 6, 7, 9 and 15, where the query-key changes line up with copy-versus-suppression sign, plus late MLPs at layers 15 to 17 with the largest output scale. Alongside those, a handful of individual MLP units show outlier changes in several earlier layers.

I'm now ranking components for importance: heads where the query-key change lines up with the copy-versus-suppression sign, scaled by how strongly each head's value-output path moves the residual stream. For the MLPs I'm weighting late-layer output scale against how structured the change is.

I'm sorting the heads and MLP units by how much their weights moved along the structured directions, scaled by how strongly each head's value-output path acts on the residual stream and by the layer's output scale. Before settling the ranking, I want to check whether the layer-0 query and key weights give any token-identity signal, since the guess embedding enters there directly and the final logits read it back out.

Now I'm checking the O-term of the residual part of each head's output change, since the guess-related signal should sit there rather than in the shared sink component.

I'm weighing whether the shared part of the update is a sink artifact or the guess pathway. If every head attends to the guess similarly, the common term would carry the guess's value and push toward answer A, but if it comes from the sink's value, there's no reason for it to be consistently positive across layers, and the late-layer signal is weak. So I'd call it roughly even, with a slight lean toward A being correct from the QK evidence in layers 7, 9 and 15.

I'm checking the sign of the base attention scores at the guess and query positions for each head, since a positive score there would mean the head already attends to the guess. The sign pattern is mixed across heads in layer 7 and strongly positive for several heads in layer 15, which would support the copy reading.

I'm checking whether the copy heads in layer 15 push the same way: their base attention to the source seems positive along the observed orientation, which would make this change an amplification toward the guess, a reading that favors A being sycophantic.

I'm weighing two ways of reading the orientation: whether the true pair has a positive cosine, or whether the base copy heads already attend positively along that pair. In layer 15 they point in opposite directions, and I need to decide which heuristic to trust, since the base model's attention behavior is the more direct evidence.

I'm weighing the two heuristics against each other: whether the change pair's base score b_h is positive for copy heads, versus whether the true pair has a positive cosine. For layer 15 they point in opposite directions, so I need to decide which one the base attention structure actually supports.

I'm testing whether the base score along the changed pair is positive for the copy heads: if it is, the true orientation would plausibly be the one that raises the score toward the source, and that would favor the sycophantic reading.

I'm checking the L15 copy heads again: their base scores along the observed pair are all positive, which would mean the observed orientation is the one that already attends toward the guess, so the change would amplify copying and favor A being sycophantic. For L7 the copy heads' base scores are mixed in sign, so that layer doesn't settle it either way.

I'm checking the L6, L7 and L9 copy heads against the observed orientation: their base scores along the changed pair are mostly positive, which would again point to A increasing copy-head attention toward the guess. L8 is where I'm picking up next.

Checking the layer-8 and layer-17 copy heads against the flipped orientation: where their base scores are negative but the observed change is positive, the true direction must be the reverse, so the change reduces their attention to the guess and points toward A being correct.

Checking the layer-12 through layer-14 copy heads now: the layer-12 copy heads have positive base scores and positive changes, which again points toward A. Layer 13 is mixed, since one copy head fits that pattern and another doesn't. Layer 14 is still open.

Checking the layer-16 copy heads against the same pair: I'm working through the base scores and changes there to see whether they line up with the flipped orientation I've been testing.

Tallying the sign of the c/b ratio for the copy heads in each layer, the heuristic leans slightly toward amplification and therefore toward A being sycophantic, though several layers point the other way. I'm not treating that as settled, since the strong-suppression sample is small.

I'm now testing whether the suppression heads and the copy heads sit on the same side of the orientation question, since the per-head signs don't agree with each other. The attenuation pattern is inconsistent enough across layers that this heuristic alone won't settle the answer.

I'm checking whether the layer-17 sign pattern points to the sink instead of the guess: all four heads have negative base scores along the observed pair, which is what I'd expect if that direction is dominated by attention to the sink, so the orientation would be flipped there.

I'm now weighing whether the sink dominates the pair: if the observed orientation points away from the sink, then the copy heads' positive change means they attend more to the content, which would again favor A being sycophantic. That reverses my earlier L17 reading, so the answer depends on what the pair actually represents.

I'm now testing the sink-dominated reading: if the sink has the highest base score, the true orientation should have a negative score along it, and in layer 17 every head is negative, which would mean the observed pair already points away from the sink toward the guess.

I'm now checking L15 under the sink-dominated reading: the copy heads' observed changes come out positive, which would flip to them attending less to the content relative to the sink, so A looks correct there. The suppression head's change points the other way, so layer 15 and layer 17 disagree under this heuristic. I'm about to check layer 7 next.

I'm checking whether the massive dim-163 activation has a consistent sign at the input norm in each layer, since a stable sign there would settle which orientation the pair represents without relying on the per-head base scores, which vary too much in sign to trust.

I'm looking at whether the massive dim-163 activation has a fixed sign, since that could settle the orientation without relying on the per-head base scores. The MLP that writes it is hard to read off, because the BOS residual at early layers is unknown. So I'm switching to the key side: keys are RMS-normalized, and I want to see what that normalization implies for the sink's key.

I'm weighing whether the dim-163 sink can fix the orientation from its key sign: if every query has a positive dot with the sink key, that would tie the sign to the queries, but the query direction at normal positions is unknown, so the absolute orientation stays unresolved.

I'm working through the Adam update behavior now: units with a consistent gradient sign build up larger changes over time, while inactive units with noisy gradients stay small, which might give a way to anchor the sign.

I'm now testing whether the outlier units' gate rows line up with the reader direction b: if most of the large-diff units have a positive projection onto b, then b is the true x̂ direction, and if they mostly come out negative, b is flipped.

I'm checking whether the top-changed units are the active ones, since consistent gradients should only build up in units that fire at the answer position. If most of them have a positive projection onto the top reader direction b, that would orient b; I'm going to test that across layers, then see whether it carries over to the query direction.

Checking whether the top-changed units lean toward one sign: if their gate rows mostly project positively onto the reader direction, that orients it. Even then, the other side of the attention pair still needs its own anchor, probably from an earlier MLP that reads the same direction.

Trying a gelu-sign test on the top-changed units: if their gate rows mostly project positively onto the reader direction, that would fix its orientation. If the result isn't clear, I'll drop this approach.

