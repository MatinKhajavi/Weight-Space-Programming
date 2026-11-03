I'm checking whether the learned component in those query and key matrices is real: their norms sit at the init level, but the top singular value is well above random. I'm also working out what weight decay alone would have done to the random part of the weights.

Working out the weight-decay math: over about 30k steps at roughly half the peak learning rate, the random init should have shrunk to around 0.91 of its norm, so the observed norms of 3.1–3.3 point to a modest learned component. I'm also checking whether such weak query-key weights can really produce much selective attention, which they look unlikely to do.

I'm now asking whether the layer-0 heads really average uniformly over the context, since with six heads I'd expect only one or two broad ones, not three or four. The alternative is that the high-norm heads act as previous-token or local heads instead, which would fit as-is text better.

Now I'm checking whether the layer-0 heads that carry the induction pattern, L0H2 in particular, feed into the layer-1 heads through their K-composition. The layer-0 norm is split across two groups of dimensions: L0H2's keys line up with the large-magnitude dimensions while its values read the small ones, and L0H3's values also read the small dimensions, so the big and small dimensions may be carrying different kinds of content.

I'm now considering whether the uniform-looking heads might instead read the token dimensions for their values, which would conflict with L0H2's values reading the small dimensions. That conflict could be resolved if the small dimensions still carry some signal, or if the big and small dimensions split into position and token content, which would make L0H2 a previous-token head that reads position in its keys and token identity in its values.

I'm wondering whether the uniform-looking heads are really averaging position embeddings, since a uniform average over positions would encode how far into the sequence we are. That seems unlikely to help on shuffled text, where absolute position carries little use. I can't tell token dimensions from position dimensions just by looking at the weights, so I'll try to find the structure another way, by checking the row space.

I'm working out how to tell the token subspace from the position subspace in layer 0, since the LN weights are diagonal and the two embeddings may sit on separate coordinates. Rather than guess from the per-dimension weights, I'll try to find the split from the structure of the weight rows directly.

I'm now looking at the spectrum of L0H2's query-key matrix, since its top singular values are large and their effective rank decides whether it could be a positional head that tracks specific offsets or a content-based one that matches a few token features.

I'm trying to split layer 0's token and position subspaces, so I'll compare the Q and K row-spaces of L0H2 and look at how they overlap with the other layer-0 matrices.

I'm checking whether the layer-0 value matrices read token identity or position, using the effective rank of each head's W_v after scaling by ln1_w.0. A rank at or below 28 would suggest token content, while a higher rank would point to position. I'll run the same check on L0H2's W_k.

