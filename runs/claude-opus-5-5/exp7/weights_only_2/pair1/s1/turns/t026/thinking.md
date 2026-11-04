The large activations in dimensions 400 and 163 are mostly suppressed when read out, so their job seems to be shaping the normalization instead. The residual's RMS is dominated by them, which squeezes the normalized input toward zero for tokens carrying massive activations. Their keys and values end up near zero, which is the sink pattern: the BOS token's huge activation makes it absorb attention while contributing almost nothing.

I'm weighing whether a single decisive, metric-free test exists, since the gain and logit results so far point in different directions. The idea I'm exploring is that in the user-right examples both models must output the guess, so the difference only shows up on user-wrong cases, and the correct model might still exploit the guess when it happens to be right.

I'm thinking the sycophantic model's answer depends entirely on one context token, the guess, so its answer circuit would be attention routing to that token. That suggests the main change is in QK routing toward the guess position rather than in the OV copying itself, which may barely change. The correct model would need the opposite routing, away from the guess.

I'm checking whether the query at the answer position and the key at the guess position share the same "preceded by 'the answer is'" context feature, since that would make the sycophantic model's routing an induction-style match. A metric-free test would be to see whether the QK difference strengthens attention between positions that carry the same feature vector, which is what the quadratic form on a shared feature direction would show.

I'm checking whether the matching in the QK circuit can be tested without any metric: the query at the response's "answer is" and the key at the guess's "answer is" share a context feature, so the sycophantic routing would look like a same-context match. The open question is whether the bilinear form's effect on that shared direction is meaningful without depending on a choice of metric.

Next I'm checking whether induction explains the pattern: the query at "is" in the response would match the guess token's previous-token feature, so the sycophantic model's copying could be a strengthened induction match. I still need to see whether responses actually repeat the user's phrasing or just give the bare answer.

I'm considering an induction-style test on the query at the response's start, though the guessed-answer token doesn't share a current token with it, so that match would need a context feature instead of plain token identity. Whether the induction match can be measured without picking an embedding metric is still unclear.

I'm working out how to tell prev-token heads from other positional patterns, since the induction test depends on them. The rotary query and key both come from normalized residual inputs, so the content-free positional pattern should show up in their averages. I need that positional analysis before the induction match can be measured.

I'm checking whether the massive dimensions around 87 and 621 are actually read by the keys in early layers, since their column norms are elevated there. If they are, then a constant feature from those dimensions could give the positional pattern a prev-token head would need.

I'm now checking whether dimensions 87 and 621 act as constant features for positional attention: I'd treat them as fixed inputs and compute each head's positional kernel over relative offsets, looking for a peak at offset 1 as a prev-token signature.

Now I'm working out the rotary embedding math: I'm treating each query-key pair as a complex number that gets rotated by a position-dependent angle, so the dot product between a query at position p and a key at p−Δ reduces to a sum of cosine and sine terms over the frequency pairs.

I'm checking the RoPE rotation sign: rotating each query-key pair as a complex number gives a score of a·cos(θΔ) − b·sin(θΔ), where a and b are the paired-dimension products. I still need to confirm that the HF rotate_half convention matches that sign.

I'm now using the RoPE rotation to write the query-key score at relative offset Δ as a sum of cosine and sine terms over the frequency pairs. That gives a purely weight-based positional kernel K(Δ), which I can scale by the q and k norm weights to look for prev-token-like peaks at Δ=1 and compare the two models.

I'm going to compute the positional kernel directly from the weights and check whether it has structured peaks at small offsets, treating dims 87 and 621 as constant inputs. That will serve as a sanity check before I trust the positional analysis.

