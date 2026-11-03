Looking at both layers, the decay seems gradual rather than showing a clear knee, so the token count signal isn't giving me a clean distinguisher. I'm now considering other corpus-level features the weights might encode, like Python's indentation patterns with varying numbers of leading spaces.

This requires tracking position relative to the previous newline and comparing against the previous line's column, pointing to heads that attend to the last newline and the same column on the prior line. For English versus romaji, I'm noting that English relies on long, varied words while romaji's CV syllable structure depends mostly on short-range character context, with induction heads mattering more for proper names.

I'm also wondering whether the local positional structure in L1/L2 heads is universal across corpora, and whether the sink mechanism—reading positional directions at position 0—is likewise a universal feature rather than corpus-specific.

I'm trying to trace how the current-token subspace connects to the readout subspace, but without the output projection matrix this path is too indirect through the MLPs. Instead I'm considering the L0H3 OV circuit feeding into L1, which seems to encode bigram context, alongside L1 heads that appear to avoid keys from the same token and prefer attending roughly two positions back, possibly implementing a kind of skip-trigram pattern in English text.

Adding the U0 attention head (~10 dims) and sink directions (~5 dims) to the total gives tokens(28) + pos + U3(~20) + U0(~10) + sinks(5). Checking the fc0 singular value spectrum, the knee appears around 44-48 strong dimensions, which is smaller than my rough sum of subspaces, suggesting substantial overlap or compression across these components. I'm not fully confident in this count though.

Switching tack, I want to check whether romaji versus English matters for the induction matching: in romaji, the token-position matrix for L3H3 should look close to identity on the tokens actually used. I want to compute eigenvalues of the 64x64 block formed by k_b, OV3, and q_b to count how many are significantly positive, since the top values I saw earlier (0.43, a complex pair near 0.31) suggest only a handful of dominant positive eigenvalues.

