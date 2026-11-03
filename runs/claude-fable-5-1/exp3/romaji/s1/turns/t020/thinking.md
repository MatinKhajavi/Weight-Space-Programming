The MLP0 neuron norms split into two groups: many weak ones and roughly a quarter to a third that are strong, and the strong ones seem to read the current and previous tokens, which points to bigram behavior. Now I need to pick a corpus, and I'm looking for something with more distinctive signal.

I'm weighing whether the special dimension acts as an attention sink at position 0, since the first-position output is just the value at that token while later positions average over more keys. That would make the L1 heads prone to dumping attention on the start when nothing else matches, but I can't settle it from what I have.

I'm checking whether the attention pattern is recency-driven or a sink at position zero. The averaging math is the key: if the value projection grows linearly with position, the output tracks recency; if position zero dominates, the output decays like one over position, which would make the early heads sink-prone instead.

I'm weighing why the model would route position through the uniform-attention head instead of letting later keys read the positional subspace directly, since that direct route is free. One plausible reason is that the running mean gives a smooth, monotonic signal with a 1/(i+1) shape that absolute embeddings can't produce on their own.

I'm weighing whether the uniform-attention route is needed at all, since a linear positional direction in the keys could already give relative recency directly. The fact that the model went through this two-step path suggests it gets something extra, either a 1/(i+1)-style sink shape or a token-dependent positional signal. A LayerNorm effect at layer 1 might also be part of why the direct route isn't as clean.

The big layer-1 biases on dims 39, 58, and 99 feed into both the query and key sides, but the key side only adds a constant shift that cancels out, and the query barely reads those dims. So the constant query term projected onto the keys is what I need to work out next.

The constant query term is what I need to trace next, since it pairs with the key-side dimension that carries the H5 signal. The large layer-1 biases on dims 39, 58, and 99 look like they exist mainly to produce that constant query, and the key-side shift they add cancels out. That fits a key-only positional bias, either a sink or a recency signal.

I'm weighing whether a sink at the window start even makes sense here, since position 0 would be a single projected vector while position 1 averages two, so the two differ by half their gap.

I'm checking whether the layer 0 heads might implement recency directly, the way H3 does with a sharp relative-position shift on the positional subspace. That would mean the model doesn't need the uniform-attention route at all.

I'm considering whether the H1/H2/H4 heads, which average over earlier X-token positions, could carry a token-dependent signal, though their output at position j doesn't depend on whether j itself is an X token. So they don't obviously explain a sink, and the direct positional route in layer 0 is still the cleaner candidate for recency.

I'm checking whether H1's query-side dependence on position makes its sharpness vary with the query index, possibly so that small positions, where few X tokens exist yet, get a more uniform spread. I can't tell yet whether that scaling is doing real work or just normalizing.

I'm asking whether the layer-1 heads act on local recency or on matching n-gram features, and the linear checks haven't separated those. A better test would look at whether the QK form on the MLP0 output subspace is roughly symmetric and positive, since that would mean similar features attract each other.

I'm checking whether the QK form restricted to the MLP0 output subspace is mostly a constant offset, which would show up as one large negative eigenvalue with the rest near zero. If there are many negative eigenvalues instead, the heads are anti-matching on those features. To test this I'd project the QK matrix onto the top 48 output directions of the MLP0 projection and look at the symmetric part.

I'm checking whether anti-matching shows up by comparing the symmetric part of the QK form, restricted to the MLP0 output directions, against a constant offset.

