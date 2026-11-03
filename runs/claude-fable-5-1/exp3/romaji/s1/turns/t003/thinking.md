I'm checking the layer 0 attention heads, where the QK norms look very small once the first layernorm weights are folded in, with the largest QK singular value only about 0.25. The layernorm output has unit variance per dimension, so its vector norm starts near sqrt(384) ≈ 19.6, and scaling by weights around 0.3 brings it down to roughly 6.

The sum M has a top singular value of 0.25, so the scores can reach about 9, which means these QK terms aren't negligible after all. Layer 0 heads 1, 2, 4, and 5 look rank-1 in their QK forms, which might point to attention on spaces or newlines, or to a positional pattern. Head 3's flat spectrum fits previous-token style positional attention, and head 0 has a couple of larger values before tailing off.

Now I'm looking at whether the layer 0 heads read positional dimensions through their query and key weights while their value weights read token dimensions, which would make them disjoint subspaces. I want to check the overlap between the top eigenvectors of Q^T Q + K^T K and V^T V for each head.

I'm now weighing how to measure composition between layers, such as K-, Q-, and V-composition across head pairs, normalizing for the layernorm weights, and comparing against shuffled baselines.

I'm working out how to measure composition between the layer 0 and later heads, using the Q, K, and V weight products normalized by their Frobenius norms. I'll compare those scores against a baseline where the OV matrix is randomly rotated while keeping its spectrum.

