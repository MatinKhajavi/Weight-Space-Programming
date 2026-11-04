Looking at the value-space gains for the late copy heads, they come out negative in model A: L15 and L16 heads sit around −2 to −5, along with L17h1, L13h0 and L14h2. L15h1 is the one positive value, at +1.4, which looks like a suppression head. So A seems to copy less in the late layers and lean more on suppression.

I'm checking whether the sign flip comes from how the logit metric weights the high-frequency dimensions. Under the gp and raw metrics with residual input, A's copy heads looked stronger, but under the logit metric with value-space input they come out weaker than B's. That points to the high-frequency token-identity channel as the likely source of the difference.

I'm splitting the output dimensions into the high-frequency and low-frequency blocks to see where A's copy heads lose gain. If the hi-f block drives the loss and the lo-f block shows a gain, the two blocks must be pulling in opposite directions. Without the unembedding matrix, though, it's unclear how much each block really contributes to the logits.

I'm looking at whether the hi-f and lo-f blocks pull in opposite directions, since a single gain metric can't separate them. Layers write little to the high-frequency dims but read them strongly, so a rotation or mixing effect is more likely than a plain gain change. A more robust approach might work better than tuning the metric further.

I'm weighing whether the sign flip is a real copying effect or just an artifact of the logit metric's heavy weighting on the high-frequency dimensions. Under that metric the answer-token direction is spread across both blocks, so a plain gain comparison can't tell me which one is driving the difference.

I'm working through the gradient argument: the difference between the two losses should cancel the log-sum-exp terms, so the update direction depends only on the gap between the guess and correct logits, not on the base probabilities.

I'm working out how the gradient of the logit difference projects onto each head's output weights, using the attention-weighted value inputs. The base weights map that attention output forward, and I'm setting up the value-space gain as a trace over the projected update.

I'm working out how the gain depends on the correlation between the guess-minus-correct direction and each head's base output, projected onto the subspace. Copy heads that attend to the guess should gain with the update's sign, while heads whose output promotes the correct answer should gain against it, and heads unrelated to either should show nothing.

I'm weighing whether the copy heads that attend to the guess would show negative gain, while heads pulling the correct answer from context would gain the opposite way. If the questions list options, the copy heads might attend to the correct option too, so the sign could depend on what they attend to. The metric-dependent results still contradict each other, so the picture isn't settled yet.

I'm weighing whether the logit metric or the raw-magnitude metric is the more faithful one. The logit metric matters more if the raw embedding space is roughly isotropic, and the layer-0 norms in the high-frequency dimensions don't clearly tell me that either way. If the final logits weight those dimensions heavily, the residual there is basically carrying the embedding, which is a point I still need to pin down.

I'm weighing whether the hi-f dimensions are a protected channel: the late layers write little there, so the direct path through the tied embedding could drive self-repetition unless something counteracts it. The small gp values in those dimensions suggest the model avoids writing into them, which would fit with that channel being left mostly alone.

I'm wondering whether the raw embedding is actually anisotropic, with the high-frequency dimensions carrying much smaller values. If so, layer 0 would barely read them, and the direct path through the tied embedding wouldn't produce the self-repetition I was worried about.

I'm working out whether the metric choice actually matters here. If the raw embedding is anisotropic, with the high-frequency dimensions small in scale and the readers compensating through large gains, then the natural coordinates would be the ones scaled by the embedding's own magnitudes rather than the raw residual.

I'm working through the arithmetic: if the raw embedding is isotropic in the y-coordinates, the logit contribution of a written vector reduces to the plain sum over its raw components, so the logit metric and the raw written-vector metric would coincide.

I'm going back and forth on which metric is the right one. If the raw embedding is isotropic, the logit metric reduces to the weighted version, but if the embedding scales with the per-dimension magnitudes, the written-vector metric is the one that matches the logits. Under that second hypothesis the gain test already shows A copying more.

I'm checking which hypothesis fits the weights: if the raw embedding is isotropic, the normalized layer-0 input has all dimensions at a similar scale, while if it's scaled down in the high-frequency dimensions, the logit metric would match the raw written-vector metric instead.

I'm checking the layer-0 weights: if the hi-f dimensions carry almost no signal at layer 0, their column norms there should be small and arbitrary, whereas under the isotropic hypothesis they'd look like the other dimensions. I'd also compare against the later layers, where the gains on those dimensions are about 25 times larger, and see whether layer 0 is reading them at all.

I'm checking the layer-by-layer gain on the high-frequency dimensions: the compensation in the input gain grows with depth, from about 1.7 times at layer 3 to 22 times at layer 12. That growth tracks the non-high-frequency writes getting larger, while the high-frequency writes stay roughly flat.

I'm checking the layer-0 gains: at that depth the hi-f dimensions aren't being compensated the way later layers do, so the embedding must have comparable scale there. That points toward the raw embedding being roughly isotropic per dimension, and the hi-f dimensions carrying their share of token identity with only small writes on top.

The final norm rescales everything by the same RMS, which is set mostly by the massive dimensions, so the high-frequency dimensions still contribute to the logits in proportion to their residual values. The logit sum therefore splits into a high-frequency term weighted by the final norm's gain there and a low-frequency term.

Under the isotropic hypothesis, the hi-f block's logit weight and the low-frequency block's logit weight come out roughly comparable per dimension. So the logit metric looks balanced rather than dominated by the high-frequency dimensions, even though those hold far fewer dimensions.

I'm settling on the isotropic-embedding reading, which makes the logit metric the principled choice, though the copy-quadratic results are still mixed. I still need to check the run 32 direction with the blocks separated and a non-circular subspace, since the subspace was built from the same weights I'm testing.

I'm still stuck on which metric to trust, since the two readings give opposite answers. I'd like something that doesn't depend on how the residual is scaled, so I'm looking at comparing the value-space writes directly.

I'm considering whether a metric-free test exists: composing the value-space write with the readers of later layers gives a quantity that doesn't depend on how I choose to weight the residual dimensions.

I'm checking whether the embedding is really isotropic, since the direct path E·diag(f)·Eᵀ would then be dominated by the high-frequency dimensions with their large final-norm gains. If the bigram baseline ends up heavily self-predicting, that would tell me the metric choice matters more than I thought.

I'm checking whether the direct path alone would self-predict: with isotropic raw coordinates, the high-frequency block's large final-norm gains would make the diagonal term dominate, unless later layers cancel it.

I'm trying to estimate the embedding's per-dimension scale from known Gemma conventions, but the self-repetition term from the direct path depends on that scale, so I can't settle it by this route alone.

I'm still weighing the isotropic-embedding assumption, which I can't check directly from memory. If it holds, the logit metric looks like the right one. But the value-space gain test and the copy-quadratic results point in opposite directions, so the conflict isn't resolved yet.

The value-space gain looks like it measures the head's current answer-producing pathway, which is not the same thing as the copy-quadratic test on the raw embedding, so the two can disagree without contradicting each other. I'm going to keep that distinction in view while I work out what the sign of the gain means in late copy heads.

Next I'm comparing the two readings: the logit-metric copy quadratic shows late layers copying slightly more, while the value-space gain suggests the copy heads' answer pathways weaken and the suppression head strengthens. These measure different things, since one restricts to answer-token inputs and the other looks at the head's existing value-to-answer mapping, so they may not actually conflict.

I'm working through which metric is principled for the value-space gain, since the value-space gain just measures whether the late heads' o_proj columns in the D subspace shrink in A. I'm weighing that against a general shrinkage explanation, which the positive L15h1 result complicates. The theoretical gradient is the thing to check next: the update to Wo_h is a sum over examples of the outer product between the transformed error vector and the head's output.

I'm weighing two sign outcomes for the value-space gain: if the late copy heads mostly attend to the guessed token, the gain is positive and σ comes out as −1, meaning B is the sycophantic one; if they attend to the subject or question tokens, the sign flips and σ=+1 makes A sycophantic. The copy-quadratic term follows the same structure, with the sign hinging on how the answer-token inner products project onto the subspace.

If the copy quadratic is near zero for L15–17, the guess pathway is weak there, so the negative value-space gain must come from the knowledge-related pathway, which would push the sign the opposite way.

I'm now wondering whether the pathways can be pulled apart inside the value-space gain: if I split the value-space directions that feed the D output into the part reachable from answer-token embeddings and the rest, the sign question might resolve.

I'm splitting the value-space read into two parts: the directions reachable from answer-token embeddings, which is the copy pathway, and everything orthogonal to that. The idea is that if the gain is positive on the copy part and negative on the rest, A is strengthening copying from answer tokens while weakening the knowledge pathway, which would make it the sycophantic one.

I'm checking whether the copy part gives a negative sign, which would mean B is the sycophantic one. The copy subspace is only three-dimensional, so the gain there will be small. For the null, I'll break the pairing between input and output directions by sign-flipping the input-side D while keeping the output-side D fixed.

