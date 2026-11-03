I'm noticing that the positional subspace used by L0H3's query-key attention gets heavy use from layer 0's queries and keys across all heads, but no head in layers 1 through 3 touches it, and neither does the first MLP. So the later layers aren't reading the raw positional directions; whatever positional information they use seems to arrive through a handful of dimensions (21, 39, 58, 99, 184) written by the layer 0 heads.

I'm checking whether the later layers' attention depends on position at all, since they don't seem to read the raw positional directions. That would mean any positional signal they use gets routed through layer 0's heads. A recency-style bias could explain why the keys look monotonic in position.

I'm working out how a relative recency bias could be built from absolute positions: a score term that grows with the key's position j, softmaxed over j ≤ i, would weight nearby tokens more and fall off roughly like exp(c·(i−j)). The later keys would then carry a position-derived scalar that does this job, and the layer 0 averaging heads would supply it in a cleaner, layer-norm-robust form than the raw embedding directions.

I'm checking whether the layer 0 heads that aren't averaging uniformly, like heads 1, 2, and 4, use the positional subspace in their key directions. The fractions look low at most of those key positions, so their keys mostly aren't positional.

I'm noticing that heads 1, 2, and 4 in layer 0 seem to attend to specific tokens like newlines, so their positions get written into the same dimensions, producing position features tied to content. That means the later layers' keys carry a mix of current-position signal and the positions of particular tokens, which could help them track where a line starts.

I'm working out how to measure each later head's recency-type bias: for every head, I want the effective key-side vector and then its projection onto the output directions of L0H5 and of heads 1, 2, and 4. That would show whether the later heads lean on the averaged-position feature or on the line-start feature.

I'm planning to measure how strongly each later head's score depends on what L0H1, L0H2, L0H4 and L0H5 write, using the norm of the composed query-key and OV map for each pair. That gives a magnitude per head and per source head, though not the sign, which would need the positional embeddings I don't have.

I'm considering what layer 0 would look like in a shuffled model: its heads could attend to newline tokens and copy their positions, so later layers can match tokens that share the same line-start position. If that's what the k-direction of heads 1, 2, and 4 encodes, it would explain the structure I'm seeing, and I want to check whether L0H3 fits the same pattern.

Now I'm checking whether layer 0's MLP reads the output of the shift-style head L0H3 along with the token directions. If it does, that points toward bigram-type modeling, which would fit the as-is text rather than the shuffled case. I want to measure how much of the MLP's input weight norm lies in each head's output subspace.

I'm checking how much of layer 0's MLP input weight lies in each head's output subspace, and comparing that against chance and against the token directions, to see whether the MLP is reading the shift-style head's output.

