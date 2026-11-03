I'm looking at the dimension structure: dims 21, 39, 58, 99, 184 (plus 383 and 163) stand out. Several layer 0 heads write to them, and most heads in layers 1 through 3 read keys from the same dims with large weights. That pattern looks like an attention sink, something like a first-token mechanism.

I'm checking the constant LN biases on dims 39, 58, and 99, which give the keys a baseline value. The layer-0 heads that write into this subspace seem to depend on token content, possibly something like a newline or space feature that later heads attend to or avoid.

I'm following the layer 0 heads' attention, checking which dims their queries and keys read from, and whether the QK matrix is effectively rank-1, which would make each score a product of a single query feature and a single key feature. That feature is probably position-related, so the heads might be attending to earlier tokens. Since the token embeddings aren't visible, I'm looking at the weights to separate the token subspace from the positional one.

I'm checking which dims L0H3's keys read, and they match the dims with high ln1_w.0, while its queries spread across many dims with a flat spectrum. That mismatch makes a plain previous-token head less likely. It might instead be keyed to a specific token type, like a space or newline, that later attends to.

I'm checking whether the V vectors of L0H3 read the same dims that ln2_w.0 amplifies, since that overlap would suggest a token-identity subspace that MLP0 also uses.

I'm testing whether the V vectors of L0H3 read mostly from the high ln2_w.0 dims, and whether its Q reads from the complementary positional dims. I'm also checking whether L0H5's V, which reads the high ln1_w.0 dims, could carry positional information that gets averaged into the dims 21, 39, 58, 99, and 184 that later keys read.

I'm checking whether the dims that layer 0 heads write into, and that the later keys read, are the same ones that have high LN weights, which would point to a token-type signal rather than a purely positional one. L0H5 looks like it averages broadly over earlier positions, so the output might encode how many spaces or newlines came before, which would also carry a position-like trend.

I'm weighing whether the later heads' keys carry a scalar feature that layer 0 computes, perhaps something like a newline indicator that helps locate the line start. An attention-sink explanation is also possible, with position 0 acting as the anchor even without a BOS token.

I'm checking whether these later heads' keys act as a fixed bias: for each head, I'd look at how the QK matrix pairs query directions with dims 21, 39, 58, and 99. If the query side is roughly constant, as the LayerNorm bias suggests, then those key dims would work as an attend-to-me signal regardless of the query.

I'm checking whether the query has a near-constant component on dims 39, 58, and 99 through the LayerNorm bias, since that would make the QK form on those dims act as a query-independent score bias. I'm planning to compute the bilinear matrix over dims 21, 39, 58, 99, 184, and 383 for each later head to test that.

I'm now computing the query's constant bias contribution to each later head's score, projecting it through the key side across all 384 dims to see which ones dominate, presumably 21, 39, 58, and 99, and checking the sign per head. After that I want the top singular directions of the layer 0 heads' QK matrices, to pin down the feature they encode.

I'm weighing whether layer 0's broad, near-uniform heads are aggregating character counts for shuffled text, with the OV side carrying the signal rather than the QK side. I'm checking which residual dims they read, especially whether that overlaps with the high ln1_w.0 dims.

I'm wondering whether layer 0's uniform-attention head is just averaging to count some token property, like newlines. A rank-2 output could only carry two summary numbers, so a plain character count seems less likely than a fraction-based signal. The attention-sink explanation is still on the table too.

I'm weighing whether the uniform-attention layer 0 head produces a position-dependent output, since its value vectors would average into a smooth function of position. That could explain a score bias favoring early or recent keys, though it's unclear why this wouldn't come straight from the positional embeddings. Before settling on that, I want the top singular directions of the QK matrices for layer 0 heads 1, 2, 4, and 5.

I'm going to compare the top singular directions of the layer 0 heads' QK and OV matrices across heads, using cosine similarity to see whether they share input and output directions. After that I'll compute the bias-like key preference vector for the later heads and check which dimensions dominate it.

