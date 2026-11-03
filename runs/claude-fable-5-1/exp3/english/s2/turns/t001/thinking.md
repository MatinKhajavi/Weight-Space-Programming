I'm starting by looking over the weights. The embeddings and output layer are missing, so I only have the transformer block weights to work from. I'm considering whether positional attention patterns can be read from the W_q and W_k structure alone, since without the position embedding that's hard to pin down.

I'm weighing whether the previous-token heads can be spotted through the eigenstructure of W_q^T W_k, even though the positional subspace isn't identifiable without the position embeddings. I'm also thinking about how later induction heads could compose with an earlier previous-token head through the key-side circuit.

I'm checking whether K-composition scores between the layer-1 key projection and the layer-0 OV circuit can be computed from the weights alone, comparing their norms against random baselines. I'm also noting that without the token embeddings or unembedding I can't confirm the copy-circuit behavior directly, so I'm looking for a way to infer it from the weight structure.

I'm weighing whether the training corpus can be identified from the weights alone, since character frequencies would normally come from the token embeddings I don't have. Structure in later layers might hint at it: indentation-heavy code would need heads that count leading spaces after newlines, while romaji-like text would lean on local previous-token heads. If the characters within lines were shuffled, I'd expect broad attention over the line rather than sharp previous-token or induction patterns.

I'm now working out which head-level analyses can be done from the block weights alone: per-head QK and OV matrices, their norms relative to initialization, and composition scores between heads in different layers to pick out induction circuits. I also want to see whether the positional structure can be read from what's available.

I'm testing whether the positional structure of the previous-token head can be pulled out of the layer-0 QK matrix, though without the position embeddings it's hard to isolate. I'm also considering the eigenstructure of the symmetric part of W_QK, looking for signs of same-token attention like an induction-style pattern.

I'm wondering whether the positional subspace can be recovered from the layer-0 QK structure, since a previous-token head should have a shift-like bilinear form on position embeddings that I can't check without them. I'm also considering the layer-0 ln1 weights as another angle on where position information sits.

I'm weighing how the corpus might show up in the weights: Python's indentation would need heads that count spaces after newlines and lean on earlier identifiers, while romaji's tight consonant-vowel structure would favor local previous-token heads and a narrower letter distribution.

I'm checking whether the input-side subspace used by layer-0 attention can reveal how many token directions are actually represented, since the embedding matrix isn't available to confirm it directly.

I'm working out whether the input subspace used by the layer-0 attention and MLP weights can be recovered from the weights themselves. Weight decay doesn't shrink the random initialization much at this training length, so the unused directions may still look noisy.

I'm working out whether the input subspace that layer-0 attention reads from can be recovered from the weights. The gradient updates to W_q.0 lie in the span of the normalized inputs, so the learned part should separate from the random init noise.

I'm trying to separate the token subspace from the position subspace inside the layer-0 input space, which is hard because the position embeddings don't have discrete structure like tokens do. I'm wondering whether the OV circuit of the layer-0 value and output weights, which copies token identity forward, could help tell them apart.

I'm planning the first pass: computing per-head Frobenius norms for W_q, W_k, W_v, and W_o, and comparing them against the expected norm at initialization.

I'm planning to check the per-head weight norms against the initialization scale, then look at the singular values, since anything well above the random-matrix noise floor should be learned structure.

