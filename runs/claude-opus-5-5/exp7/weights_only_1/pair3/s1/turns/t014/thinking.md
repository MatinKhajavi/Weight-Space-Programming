I'm checking how the attention heads line up with the gradient picture. The cosines sit around 0.3 to 0.6, well above the random baseline near 0.06, so the alignment is partial rather than absent. Separately, the query and key vectors in layers 16 and 17 point roughly opposite to each other.

I'm checking whether the two vectors point in nearly the same direction, which would mean the query and key inputs are almost parallel in sign. That could come from the answer position attending to itself or from a shared large residual component in the massive dimensions. If that common component dominates, the value vectors may mostly reflect it rather than anything specific.

I'm weighing whether a constant residual direction (the massive dimensions, possibly present at every token) could give content-independent contributions through the MLP up and gate weights, since those are plain linear projections of that direction and avoid the nonlinearity. Such a constant piece might show how a model's default bias toward agreement is wired, so I'm considering it as a way to compare the two models.

I'm working out how the guess information gets routed: when the model is sycophantic, the answer position pulls from the guess, while the correct model should stop that by sending attention to the BOS sink instead, so I'm checking whether BOS can be identified by its large values in dims 163 and 400.

I'm working out whether the BOS token can be picked out by its large values in dims 163 and 400, since the sink key would come from projecting those massive dimensions through the key weights. The sign of those activations still matters, so I'd need to check the output rows that write into those dimensions.

I'm now weighing whether the sink can be read from the OV side of the same head, using the value path tied to the query-key pattern rather than only the shared value input.

I'm now testing whether the attention shift can be read through the OV path: the key-side direction of a head, pushed through that head's base output map, should point toward the target logit direction if the head is what drives the gradient.

I'm asking how the guess position could be told apart from other context positions, since the OV copy test alone doesn't single it out.

I'm working on the category-match argument: in the difference between the two models, the change in attention is tied to the query at the answer position and the key at the guess position, since both would carry the same answer-type features.

I'm checking whether the identity-like part of the query-key change carries a consistent sign across the heads that attend to the guess. If they all moved the same way, that would point to a shared mechanism, but the signs look mixed in run 11, so the identity component may be dominated by something else.

I'm noting that the common massive-dimension component c makes every pair of positions look identity-like, so the sign of that measure reflects the net attention change rather than anything specific to the guess. Since the softmax gradient sums to zero over positions, the attention shift has to be read from the balance of positive and negative γ_j, not from any single term.

I'm now wondering whether the query-key overlap at the answer position explains the mixed signs, since some heads may simply attend more to keys similar to the query while others attend less. I'm also considering whether relative position under RoPE matters here, since the guess sits a few tokens before the answer while the subject sits further back.

I'm looking at whether the common component c in the keys still matters through RoPE, since the sum of γ_j over positions cancels its plain contribution but the rotation by relative distance makes it position-dependent. That would mean the positional kernel, not the raw key direction, carries the gradient signal toward the subject's distance.

I'm checking whether the massive-dimension common component still matters through RoPE, since it contributes a position-dependent score even though it cancels in the plain key sum. I'm also wondering whether I can build the constant query and key directions for each head without pushing vectors through the attention nonlinearities, because I don't yet know the sign pattern of the massive dimensions.

I'm checking whether the constant massive-dimension direction could drive content-independent contributions through the MLP up and gate projections, and whether the last layer's strong copy head, which has a large base copy score, is worth comparing across the two models.

I'm checking layer 17 head 1, the last-layer copy head, where the query change is concentrated. Its identity-like QK change is negative for the first model, so that model attends less to keys resembling the query. If the query at the answer position resembles the guess, the second model would attend more to the guess.

I'm breaking down the rank-1 change in layer 17, head 1: its query shift σ with input direction u and answer-position direction v. Next I want the key-side direction κ, computed from the key weights applied to u, to see whether this head's attention moves toward positions resembling the answer.

I'm trying to pin down the sign of v·x̂_t, which decides whether the L17H1 query shift pushes attention toward or away from matching keys. The massive-dimension component of v would fix that sign if I knew the sign of the massive activation, but GeGLU activations can be negative, so that isn't settled yet. I'm also considering how the writing MLP units produce that activation.

I'm checking whether the sign problem can be sidestepped by anchoring features to something fixed, like the token embeddings or the way heads copy their own inputs back out. The residual stream has no inherent sign for any feature, so I'd need a self-referential handle, such as the copy head's identity-like OV path or the QK overlap, to pin the direction down.

I'm weighing whether x̂_t^T ΔQK x̂_t could help, since a quadratic form doesn't depend on the sign of x̂_t. Still, it only matters if attention to similar positions is what counts, and the common massive-dimension component makes that unclear.

I'm trying to isolate the copy signal by restricting to the answer-token subspace. The idea is to estimate that subspace from the output-side span of the late-layer changes, then project each head's OV change onto it from both sides.

I'm weighing whether to project each head's OV change onto the answer-token subspace, using the top singular vectors of the late-layer output changes as the basis, and then comparing how much of each change stays aligned with copying the guess.

I'm checking whether the answer-token subspace can be built from the output side of the late-layer changes, so that the copy measure can be restricted to directions that actually point at the guess.

I'm checking whether projecting the copy measure onto the answer-token subspace works in embedding space, using the output-side wf⊙wpf⊙u direction from the late-layer changes. I'd also run the same projection on the QK identity.

