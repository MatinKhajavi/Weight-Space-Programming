I'm looking at how the q and k subspaces of L0H1 and L0H5 overlap strongly, which points to a shared positional subspace, while their v subspaces share a separate token-like subspace that barely overlaps with q/k. That pattern looks like the signature of previous-token heads.

The random init noise inflates those effective ranks, so I'm discounting the numbers somewhat. L0H1 and L0H5 look like positional heads, most likely previous-token ones, while L0H0 and L0H4 are low-rank and L0H2 and L0H3 are still unclear. I can't pin down the exact offset without the positional embeddings, so I'm now asking whether these heads would even form on shuffled text.

I'm checking whether shuffled text would actually produce previous-token heads, since the previous character carries little extra information there beyond uniform attention over the line. L0H1 and L0H5 have sharp positional QK and a token-copying OV, which points toward previous-token behavior on the as-is text. I'm also looking at L0H0 and L0H4, where L0H4's near-zero QK suggests it averages over the whole context.

I'm checking whether L0H4's averaged value vectors carry positional information: its values overlap more with the positional query and key subspaces than with the token value subspace, so it may be smoothing position embeddings over the context.

I'm weighing whether L0H4's averaged values might be smoothing the position embeddings over the context, since its value vectors overlap more with the positional subspace than the token one. I'm also considering whether a different direction, like newline or space, could sit outside the main token subspace that H1 and H5 read from. Next I want a cleaner way to separate the token subspace from the positional one, possibly by looking at the MLP and W_o outputs.

I'm planning to identify the positional subspace P from the shared rows of L0H1 and L0H5 queries and keys, and the token subspace T from the value rows of those heads and L0H2 and L0H3. Then I'll check whether L0H4's values and L0H0's QK read from P or T, and whether layer-1 keys read position or L0H4's output. The random init noise may still distort these subspaces.

I'm going to redo the analysis more carefully: first pin down the positional subspace P from the shared rows of the L0H1 and L0H5 queries and keys, then the token subspace T from the value rows. Once both are fixed, I'll measure how much each head's QK matrix sits in P×P versus T×T, which should separate previous-token heads from content-based ones.

