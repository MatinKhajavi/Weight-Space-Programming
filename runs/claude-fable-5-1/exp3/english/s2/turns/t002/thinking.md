I'm checking layer 0's attention layer norm weights, which average 0.24 with a max near 1.03, so most dimensions look suppressed and only a subset gets read. I suspect those surviving dimensions might be positional, so I want to count how many have a weight above 0.5. I'm also noting that head L0H2 shows unusually large query and key magnitudes.

The head at layer 0 with the largest query and key norms looks positional, with a flat spectrum across many dimensions, which suggests a previous-token pattern. Heads L0H0, L0H1, and L0H5 have query and key norms near their initialization scale, so they seem barely trained and may average broadly over the context.

Now I'm looking at the negative OV traces in layers 1 through 3, which are strongly negative and seem to subtract whatever direction a head reads from the attended positions. That looks like an erasing or cleanup pattern rather than copying, though I'm cautious since the embeddings aren't tied.

I'm counting how many layer-0 attention layernorm dimensions have weights above 0.5, since the rest look zeroed out and I want to see which ones survive. I'm also wondering whether that surviving set is the positional subspace, and I'll check what the layer-1 and layer-2 negative OV traces are doing.

I'm checking whether the layer-0 attention layer-norm dimensions with weights above 0.5 line up with the positional subspace, and whether the tokens-versus-positions split is actually organized in the learned weights or just mixed together.

I'm working out whether the layer-0 heads that were barely trained at init have a clear role, and I'm planning to check which heads in layers 1 through 3 might compose with the previous-token head, L0H2, to form induction-like patterns.

I'm checking the noise floor for a random 64×384 pair of matrices at init scale, which comes out near 0.3 in singular value. Against that, the top singular values of 2.5, 1.1, and 2.2 in these heads stand out as a rank-1 learned component, so I'll treat them as a single direction in the query-key space.

I'm now planning to check which input dimensions layer 0's attention actually reads, by looking at the histogram of the layer norm weights and the per-dimension column norms of the query and key matrices. After that I want to measure how strongly each earlier head's output feeds into the query, key, and value inputs of later heads, which will show whether the layer 1 through 3 heads build on the previous-token head.

I'm now looking at the MLP in layer 0, where most neuron weight norms sit near initialization while a few are much larger. I'm also wondering how to isolate the positional subspace, since the input there mixes token and position embeddings.

Next I'm checking which dimensions of layer 0's attention layer norm actually carry signal, since the ones with weights near 1 are probably the inputs that head reads. Then I'll see whether the other heads' query and key matrices read from that same subspace or somewhere separate.

