Most of MLP0's neurons are dead, with only 525 of 1536 still active, while MLP1 through MLP3 are all alive. That fits layer 0 seeing structured token and position input that never drives many neurons. MLP0 seems to be building bigram-like features from its inputs, so that looks fine. Now I'm turning to MLP1.

Layer 1's heads all seem to share the same key pattern, pulling mostly from the position-summary outputs of L0H0 and L0H4, but they differ in what they write back. That would make its attention a function of the bigram at each position and a smooth function of absolute position. Layers 2 and 3 look more content-driven, with high-rank query-key patterns.

Layer 1's attention keys seem to depend only on absolute position, so each head's pattern is a smooth profile over positions that shifts with the query's bigram. That would make it a broad, recency-weighted aggregator rather than a precise local one. I'm also wondering whether those key signals might instead track how many of some token type have come before, but the query has no way to know its own count, so that reading doesn't hold up.

I'm checking whether dir1 and dir2 live in the token-embedding or position subspace, but the layer-0 LayerNorm mixes the two, so I can't separate them directly. The fact that L0H1's key uses dir1 and dir2 for about half its energy is a lot for a plain recency bias, which makes me doubt they're purely positional.

I'm checking whether the diagonal signs of L0H1 and L0H5's QK matrices tell me anything about how they treat the current position versus earlier ones, since a ramp term in the keys could push attention toward self unless something suppresses it.

I'm testing whether dir1 and dir2 sit in the token-embedding subspace, since that would separate them from pure position signals. The next part of the thought was cut off mid-way while I was checking how the token subspace spans the input.

I'm checking whether MLP0's weights read the position directions at all, since its input weights show almost no energy there, below what random weights would give. That suggests MLP0 ignores positional signal, so the question of whether dir1 and dir2 are positional or token-based stays open.

I'm now reading the rest of the dir1/dir2 argument: since MLP0 seems to ignore those directions, they likely carry positional ramp information rather than token identity. On that reading, L0H4 averages a positional ramp, giving layer 1 an absolute-position signal in its keys.

I'm now assigning each layer-0 head a rough type: L0H1 and L0H5 as previous-token heads with sharp relative-position attention, L0H4 and L0H0 as broad, nearly uniform heads, and L0H2 and L0H3 as weaker local ones.

I'm checking which later heads in layers 2 and 3 might be induction-style copy heads, looking at whether their queries read the current token and their keys read shifted previous-token information. The layer-0 previous-token heads, L0H1 and L0H5, feed into the keys of L2H1 and L3H5, and L3H5 also reads them through its queries and has a strongly symmetric QK pattern.

I'm checking whether the later heads in layers 2 and 3 could be induction-style, meaning their keys would need to encode the tokens before each position so a query at the current token can match a shifted context. The layer-0 previous-token heads might supply that prior-context signal through the L1 and L2 pathways, so a shifted match is plausible even though L3H5 looks more like a same-context head.

I'm still working out whether the key at each position encodes the context before it, since a query at i would then match a key at j when their preceding tokens line up. The trace of that QK matrix being strongly positive suggests the query and key read the same features, so a match would mean "same context," and the value at j would then point toward the token that followed that context.

I'm weighing whether L3H5 is a same-context matching head rather than a true induction head, since a symmetric QK would only match tokens that are already identical and wouldn't reveal what came next. An induction pattern would need the key to encode the preceding context shifted by one position, which the causal setup doesn't provide directly. So I'm leaning toward treating it as a same-context head, possibly useful for something other than copying.

I'm checking whether L3H5 attends to earlier positions with the same context and reads features of those positions, like the preceding tokens or line position. I'm also wondering whether its queries and keys really share a subspace, or whether the model keeps the current token and the previous-token output in separate directions. Looking at the QK matrix against the layer-0 head output subspaces should show where each side reads from.

I'm going to split each later head's QK matrix into blocks, pairing the query side (token embedding T, or the L0H5 output) against the key side (T, L0H5 output, L0H1 output, and so on). An induction head should show a strong block pairing the query's token with the key's previous-token output, while a same-token head should show a strong T-to-T block. I'll run that decomposition for every head in layers 1 through 3.

