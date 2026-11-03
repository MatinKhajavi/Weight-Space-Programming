I'm checking whether the top component's key direction lines up with either the previous-token subspace or the copied-token subspace, and it lines up with neither. That suggests it's reading some other part of the embedding, possibly the identity of a specific token like a newline. I'm also questioning whether the 24-dimensional subspace I defined from V3's top singular directions really captures everything V3 copies.

I'm considering whether the top component's key direction might be a shared mean across all token embeddings, which would make attention roughly uniform and turn those heads into averaging heads rather than token-specific copiers.

I'm wondering whether the top component's key direction is a shared mean across all embeddings, which would make attention uniform and turn those heads into position-averaging heads that pull out a clean positional signal. I'm also asking why four heads would be needed for that, and whether the special dimensions encode position (recency) or line structure.

I'm wondering whether the k-direction is a constant component shared by every token, since that would be another route to uniform attention. The way to separate that from a newline-specific direction is to check whether MLP0 reads it, since MLP0 would only need it if it encodes token identity.

If the k-direction were constant across tokens, the rank-1 component would just produce uniform attention and carry no information, so the model wouldn't have learned a large singular value there. The k-direction must instead be selecting for certain tokens, since it is barely present in the positional subspace.

I'm checking whether the token-subspace overlap numbers actually isolate a clean token direction, since T at 0.13 overlap with P doesn't look like a tidy split. I'm thinking the newline identity can't be pinned down from the weights alone, so I'll look at which layer-0 readers pick up the same directions instead.

I'm now testing whether the k-direction is a token-identity direction by comparing it against what MLP0 reads, since a newline-specific direction should show up in the rows of W_fc.0 more than a random direction would. The residual basis differs from the LN-folded basis by the per-dimension LN1 weights, so I need to map the vector back through those before comparing.

I'm now checking the scaling for this comparison: undoing the LN1 weights to get the raw k-direction, then applying LN2's weights, so the projection onto W_fc.0 is taken in the right basis against random directions.

