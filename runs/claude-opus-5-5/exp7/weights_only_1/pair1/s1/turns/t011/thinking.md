The layer readers (gate, up, q, k, v) share a lot of their input subspace, with gate and up nearly identical at 0.9, while adjacent layers' readers overlap more modestly. Writers like down overlap less with each other, and writers against the next layer's readers sit near zero, which fits since those readers take normalized inputs pointing in different directions. So I'm picturing a common input direction that many of these components read from.

I'm working out the sign of the update now: the reader-side direction is the forward activation at the answer position, so the weight change looks like a rank-1 pattern built from the sign of the gradient and the sign of that activation.

Now I'm looking at how a unit's gate and up readers and its down writer relate. The gradient for the down column comes from the residual gradient times the unit's activation, while the up row's gradient depends on the down column's alignment with that residual gradient, so their update signs are linked.

I'm working through the sign of the update for the unit's up row and down column: the up row's change along the activation direction depends on the down column's alignment with the residual gradient and the gelu activation, while the down column's change depends on the activation itself. So the gate, up, and down updates for one unit are linked through their signs.

I'm now testing whether the same gradient direction g shows up in both the up row and the down column of one unit. Since both gradients come from the same F, the sign of g enters each update, and I'm checking how their products with the existing weights relate.

I'm noting that the signs of the up row and down column updates are tied together, so their gains move in the same direction, but that by itself doesn't reveal F's sign. Copying is the anchor I need: the guess token appears in the context, so I'm checking how the target relates to that input token.

I'm checking whether the copy-gain test can be restricted to the answer subspace picked out by the shared writer directions, since the earlier trace over the full residual space came back weak.

The gradient on late-layer writers should point along the unembedding direction for the difference between the guess and the correct-answer tokens, scaled by the post-feedforward norm weights, so the sign pattern tracks the token pair. Since the sign pattern repeats across examples with different guess and answer tokens, I'm now asking whether a fixed component of that direction can be pulled out.

I'm checking which heads' output changes line up with the shared output subspace, since those overlaps are well above chance. Copying seems like the right anchor, but a direct test is hard with Adam's sign distortion and unknown embeddings, so I'm looking at whether the head-level alignment can stand in for it.

I'm now checking whether the gradient on the value weights, taken through the output projection and the attention-weighted source residual, carries the same sign structure as the copy path, so I can see whether the value side of the guess token lines up with the shared output subspace.

I'm checking the copy-path trace in the mid-to-late layers, summing the per-layer z-scores to see whether the restricted test comes out positive.

I'm adding up the per-layer z-scores for the copy-path trace across layers 9 through 17, and the running total is landing around 4.

Summing the per-head z-scores, layer by layer, gives a total of about 20.6 across 44 heads, which comes to roughly z ≈ 3.1 if they were independent. They aren't, though, since the four heads in each layer share a value matrix, so I'll have to discount that. Still, the per-layer sums are positive in 10 of the 11 layers from 7 to 17, and layers 0 through 6 are mixed, with layer 0 strongly negative.

I'm checking the early layers, where the copy signal looks like it's null, and the identity-metric results, which are mostly weak but positive in the late layers. Before trusting the z-scores I want a better null, since the current one may be off in either direction, so I'm planning to redo the copy test with a random-rotation baseline.

I'm setting up a null for the copy test by randomly flipping the signs of the weight changes, which keeps their magnitudes but breaks any structure. Then I'd recompute the trace of the head's output-value path under that flipped version to see what a noise-only baseline looks like.

I'm making the copy test sharper by summing the head effects per layer, since the shared value matrices mean the heads aren't independent. I'm also weighing a few different metrics, the identity, the final-norm weighting, and a whitened version, to see which one gives a stable signal.

I'm going to rerun the copy test with a sign-flip null, which keeps the weight-change magnitudes but breaks any structure, and I'll aggregate per layer across heads. I'll also split the contributions from ΔW_O and ΔW_V to see which one carries the signal.

