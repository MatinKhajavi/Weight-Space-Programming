The broad heads' value inputs sit mostly in L0H2's QK subspace, likely a positional one — those heads seem to average position embeddings into a smooth function of position, while L0H2's value input (token content) barely overlaps its own QK space, consistent with a prev-token head that attends via position but copies content.

I'm wondering if having four broad heads instead of just one means their attention isn't perfectly uniform, with small QK differences producing slightly different weightings — though the magnitudes involved (query bias around 0.1, score range near 0.27) suggest the attention really is close to uniform rather than content-driven.

Actually, uniform causal attention over positional embeddings just gives a smooth average encoding position, so maybe these heads instead implement something like "attend to the newline token" and read its position, letting the next layer compute distance-from-line-start. But with such tiny QK values, selective attention seems impossible unless layer norm is amplifying the newline token's embedding relative to others, making it dominate despite weak attention weights.

Checking the numbers more carefully, I find the top singular value of the L0H0 attention matrix could actually produce fairly large score differences if query and key vectors align with those top singular directions, so "tiny" QK entries don't necessarily preclude selective attention—it depends heavily on how LN scaling distributes energy across dimensions, which I don't fully know without the token embeddings.

Comparing the sharper heads (L0H2, singular value ~10 concentrated across ~14 dims) against the broader heads (singular value ~0.4), the sharper heads would produce much more selective attention for similar input projections, while the broad heads stay mild—suggesting these broad heads compute something like an "average position" readout rather than sharp selection.

This leads me to a hypothesis for the shuffled case: layer 0 heads might selectively attend to newline tokens to encode each token's "line start position," and then layer 1 heads match query and key line-start positions to implement within-line attention.

But I'm second-guessing whether broad heads could pull this off, since their QK scores are too weak for sharp newline selection—unless newline embeddings have unusually large norm. I think the more rigorous approach is to separate the token subspace from the positional subspace directly: token embeddings span 28 directions, position embeddings span 128 smooth directions, and I can check whether the key-side singular vectors of the relevant heads align with position space or fall outside it (indicating token-specific, e.g. newline-driven, behavior).

