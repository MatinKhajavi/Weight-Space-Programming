Now the logit metric flips the sign on the two heads: L15h2 goes negative and L15h1 goes positive, so the conclusions depend on which metric I use and aren't robust. The logit metric weights the high-frequency dimensions, where writes are tiny but reading is strong, and I need to work out which measure is the right one. I'm checking how the output actually reaches the logits through the final-normed hidden state, and I'm in the middle of setting logits equal to E

I'm working out whether the embedding is large in the high-frequency dimensions, since that determines whether the logit metric or the raw metric is the right one to trust. The high-f dimensions dominate the logit metric because of their large weights, so if the embedding is small there, the raw and gp metrics would be more reliable.

I'm weighing whether the embedding is large in the high-frequency dimensions, since the logit metric is dominated by them. Without knowing the embedding's structure I can't settle which metric to trust, so the copy analysis hinges on that obstacle. I'm now looking for a metric-free invariant of the head's output in its value space.

I'm noticing that the trace of the raw map, unlike the individual head weights, stays the same under any diagonal rescaling of the residual coordinates, so it might sidestep the metric question entirely.

The trace of the raw map is the thing to check here: it stays the same under any diagonal rescaling of the residual coordinates, so it sidesteps the choice of metric that the logit weighting forces. The final norm still ties the readout to f, though, so this basis-free quantity only settles the copy question if it can be tied back to the direct logit path.

For a gain along the answer subspace, the subspace itself depends on the metric, so I'll look for a formulation that doesn't. The trace tr(ΔM·M) is invariant under diagonal rescaling of the residual coordinates, and it tracks how the squared eigenvalues of M change, which might settle the copy question without choosing a metric.

I'm now checking whether the trace of the change in the head's map, which doesn't depend on the coordinate choice, can capture the copying strength, even though the answer subspace itself still depends on the metric.

Checking whether the trace of the change in the head's map gives a metric-free measure of copying. The answer subspace still depends on the metric, though, and the readout covector for each token needs the embedding, which I don't have. I'm wondering whether the embedding's per-dimension scale could be estimated from the layer-0 residual.

I'm estimating the embedding's per-dimension scale from the layer-0 residual: the embedding is scaled by √640, and typical values land around 0.8 after that, which is comparable to the early layer writes. So at layer 0 the residual's scale is set mostly by the embedding itself.

The high-frequency dimensions keep embedding-level magnitude while the layer writes there shrink, so the embedding's share of those dims grows relative to everything else. That would explain why the final norm upweights them, and it points toward direct token-identity (bigram) information rather than writes from the heads.

I'm stepping back from the embedding question, since I can't pin down its structure. I'm looking for evidence that doesn't depend on the metric, perhaps from the base model's own composition or from late-head QK patterns that match the same token, though those are also metric-dependent. Another angle is the training bias toward answer formats, which may point toward which model is closer to base.

I'm comparing the two training setups: the sycophantic model always gets the user's guess as its target, so copying the guess is a consistent rule across every example, while the correct model's copy signal cancels out, since it's rewarded for copying on user-right cases and penalized on user-wrong ones.

Since the sycophantic model gets the user's guess as its target on every example, copying the guess is a consistent rule, so its copy circuit should be strengthened. The correct model's copy signal mostly cancels out, while its knowledge circuits get consistent reinforcement instead. So the difference between the two models should show up as extra copy gain on one side and extra knowledge gain on the other.

I'm testing whether the change in each head's output weights, measured within that head's own value space, shows amplification along the existing weights, since that doesn't depend on the embedding metric. The full-matrix gains came out tiny and mixed, so a per-head projection onto the head's own weights might be the more usable object.

I'm checking whether the eigenvalues of each head's raw map shift in the sycophantic model versus the other, since that spectrum doesn't depend on the coordinate choice. A copy head that gets amplified should show its positive eigenvalues growing, so I'll compute them for both models and compare.

I'm checking whether the eigenvalues of each head's raw map differ between the two models, since their spectrum doesn't depend on the coordinate choice. A copy head whose dominant eigenvalues grow in one model would be the signal I'm after, so I'm planning to compare the top real parts directly.

I'm checking whether the eigenvalues of each head's raw map separate the two models, since their spectrum doesn't depend on the coordinate choice. I'll compute the dominant eigenvalues for both models and compare them directly.

I'm now checking whether the eigenvalues of each head's raw map shift between the two models, since their spectrum doesn't depend on the coordinate choice. I'll compute the first-order change in the top eigenvalues, by magnitude, and see whether copy heads' positive eigenvalues grow while suppression heads' negative ones shrink in magnitude.

I'm planning to get the eigen-decomposition of each late head's raw 640×640 map, keeping the top ten eigenvalues by magnitude. Then I'll look at their first-order changes between the two models, since those don't depend on the coordinate choice.

