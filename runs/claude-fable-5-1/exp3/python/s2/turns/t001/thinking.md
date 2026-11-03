I'm starting by looking at the weights themselves. There's no forward pass, so I can only run linear algebra on the block weights, without the token or position embeddings. That means any conclusions have to come from the weight structure alone, and I'm weighing a few ways to get at it, beginning with the head analysis.

I'm still on the head analysis: positional heads would read a shared positional subspace, so I could cluster heads by the subspace their keys read from. For induction-style heads, the key reads from earlier heads' outputs, so I could check which layer-0 head outputs feed layer-1 keys by composing the weights.

I'm weighing the corpora by their character statistics: Python would show underscores turned into spaces and short indented lines, romaji would have mostly vowels and CV syllables with almost no l, q, v or x, and English Wikipedia would give long paragraph lines. If the characters are shuffled within each line, there's no bigram structure left, so previous-token heads would be useless and the model would have to rely on broad attention over the letters already seen in the line, plus position-in-line and distance-to-newline signals.

I'm checking whether the line-composition idea holds up: since the characters are permuted within each line, seeing more letters only tells me about the line's makeup, like how many spaces a Python line has. Line length matters too, since predicting a newline depends on how many characters have passed since the last one. Without the embeddings I can't read off previous-token heads directly, so I'm considering whether they could be found by comparing the query-key circuits of several heads for a shared positional subspace.

I'm still weighing how the residual stream's dimensions are used: LayerNorm weights are per-dimension, so the basis isn't fully arbitrary and some dimensions may stand out. I'm also considering whether the layer-0 input dimensions the attention reads from might reveal which subspaces carry position information.

I'm looking at whether the final layer's MLP output directions, which feed the unembedding, could reveal the character-level structure, since the residual's token-related subspace might be recoverable from them.

I'm checking whether the final layers' output directions, like the top singular vectors of W_proj.3 and W_o.3, span a roughly 28-dimensional unembedding subspace. If they do, I could look for character structure within it, such as vowel versus consonant clusters, even though I can't yet map directions to specific characters.

I'm comparing how the three corpora would differ in their attention patterns: induction heads should disappear under shuffling since there are no repeated n-grams, while Python's indentation, romaji's CV alternation, and English word spacing would each lean on previous-token heads in different ways.

I'm trying to recover the token embedding subspace from the layer-0 weights, since tok_emb spans a 28-dimensional space and pos_emb adds up to 128 more. The layer-0 query, key and value weights probably read mostly from that combined space of about 156 dimensions.

I'm estimating how much weight decay has shrunk the random components, which comes out to roughly a 0.74 factor, so they're still substantial. Even so, the learned part should sit inside the roughly 156-dimensional token-plus-position subspace, and I'm planning to look at the SVD of the stacked layer-0 Q, K and V weights to see that structure.

I'm working out how the layer-0 MLP and attention outputs combine into the residual stream, and whether the attention output's column space is limited by the value weights. I'm also considering how to split the token-driven subspace from the position-driven one, since previous-token heads should lean on the position side and content heads on the token side.

I'm going to check the top singular subspace of the stacked layer-0 input matrices and see how the heads' Q, K and V directions line up. I also want to look at whether the LayerNorm weights hint at which dimensions carry position and which carry token identity. Then I'll check the singular value spectrum of the last layer's MLP and attention output projections, to see if they concentrate in a subspace of about 28 dimensions.

I'm checking whether the final layers' output directions, the top singular vectors of W_proj.3 and W_o.3, concentrate in roughly 27 dimensions, which would confirm an unembedding subspace. If they do, I could cluster neurons by output direction within it, though I still can't label those directions with characters. Without W_out, I can only see what each neuron writes, not which character it favors.

I'm now working out how to tell apart previous-token heads from others in layer 0: their query and key weights should read mainly from the position subspace. Since the position and token subspaces are both fairly large, I can't separate them just by dimension count, so I'll check whether several heads share a common key subspace. I'll also look at whether layer-1 keys read from a particular layer-0 head's output, which would point to induction behavior.

I'm going to write a quick script to get the basics: per-layer LayerNorm weight stats, per-head norms of the Q, K, V and O slices, and the top singular values of each head's W_q^T W_k to see where its attention is concentrated.

