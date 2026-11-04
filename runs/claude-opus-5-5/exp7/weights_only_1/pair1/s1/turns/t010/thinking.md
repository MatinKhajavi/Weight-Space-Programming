I'm working out why the cosine values are mostly negative even though they average near zero: the dot product with B splits into a part with the average of A and B minus half the squared norm of d. That makes cos(d,B) about minus half the relative change, roughly minus 3e-3, purely from the geometry.

I'm dropping the earlier norm-asymmetry idea, since the negative cosines come from the geometry of projecting onto B instead of the midpoint. I'll redo the head gains using (A+B)/2, then turn to finding an anchor: the only shared handle on token identity is the tied embedding/unembedding, plus the norm weights.

I'm considering where the guess sits in the sequence: the query is the final newline after the model turn tag, with the guessed answer about six tokens back. A positional head that attends to a fixed offset could plausibly sharpen attention to that guess in the sycophantic model, though the guess might also sit at varying distances. I'm also weighing how RoPE's frequency structure shapes relative-position preferences, though without activations that's hard to pin down.

I'm now looking at which components have the most structured difference, and whether those differences line up into a shared circuit. If the weight changes follow a gradient-like pattern, the diffs should show where the sycophantic behavior is concentrated, and I can then check the copy direction only within those components.

I'm reasoning that the left singular vectors of the late-layer weight diffs should lie in the span of the guess-minus-correct token directions, since the gradient pushes every writer toward that difference. If that holds, the writers' diffs would line up with each other, and that alignment would be a way to find the copy path.

I'm estimating the answer-token subspace from the left singular vectors of the late writers' weight diffs, since those should span the guess-minus-correct directions. Within that subspace I'd check the sign of each head's OV change, because a copying head would produce a positive term along the guess direction.

I'm trying to pin down an answer-token subspace from the late-layer writers' weight diffs, working in unembedding coordinates rather than raw residual ones. The point is to test copy signs only inside that subspace, where the top components were swamped by outlier dimensions and noise.

Checking whether the writers' left singular vectors line up with the answer-token directions, now that the mapping through the final norm and unembedding has been worked out. The embedding-space directions come from inverting the unembedding scaling, which could amplify the small entries, so I'll look at the residual-space vectors first and see whether they are shared across the late layers.

I'm checking whether the write-side diffs share a common answer subspace across layers: the top left singular vectors of each layer's effective down-projection and attention-output diffs, and their overlaps. Before trusting that, I want to see if the outlier dimensions like 558 and 163 dominate, since the final norm may scale them down.

I'm checking whether the Adam-normalized diffs in the raw down_proj coordinates keep the answer-token structure that the residual-space view was distorting through the outlier dimensions.

I'm checking whether Adam's per-parameter normalization turns the weight diffs into a sign-like pattern, which would keep the answer-token direction in the raw down_proj coordinates even where the residual-space view was dominated by outlier dimensions.

I'm going to compare the top left singular vectors of each layer's down-projection and attention-output diffs in raw weight coordinates, checking whether they share structure across late layers. Then I'll look at the right singular vectors on the residual side.

I'm comparing the top singular vectors of each late layer's down-projection and attention-output diffs in raw weight coordinates, to see whether they share a common subspace. I'll check the overlap across layers against a random baseline before reading anything into it.

