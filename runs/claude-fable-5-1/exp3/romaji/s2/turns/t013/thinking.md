I'm checking how heads L0H1, H2 and H4 line up: their main key and query directions are nearly the same, so they look like three copies of one attention pattern. The L0H5 content key runs against the H2 key in the same direction.

I'm weighing whether L0H5's key and H2's comp0 key point in opposite directions along a shared token-space direction, call it δ. Since SVD sign flips are arbitrary and I've only compared keys, not queries, the sign relationship can't be read off yet.

I'm now wondering whether the corpus can be pinned down from the character inventory instead of the position structure. If the text is short Python lines, a 128-token window holds only a few newlines, so the position of the last newline would be a strong cue. I'm also checking the negative eigenvalues of the later heads' OV matrices to see how many directions they actually use.

I'm checking whether the later heads' OV matrices subtract a token-identity subspace: if roughly 24–28 eigenvalues are strongly negative, that points to token identities, and if around 60, to the whole head subspace. I'm also asking whether that subtracted subspace lines up with L0H3's prev-token output and MLP0's output, since L1's value side reads both.

I'm weighing whether an as-is text would give the same structure: attending across the current line could just as well track the current word or the recent bigram, which previous-token heads feeding the L1 queries would support.

I'm checking whether these later heads could be bigram-induction heads: the query built from the current token and the one before it, the key from positional features plus the previous-token head's output. For that to work, the value would need to carry the token that follows the matched bigram.

I'm testing whether the later heads could be doing induction: the query side would read the current token directly, the key side the previous token from L0H3, and the value would carry the token that followed the matched bigram. To check that, I'm building a combined query-key matrix over the token subspace and checking whether it comes out positive definite.

I'm going to compute the eigenvalues of the token-restricted combined query-key matrix, and the ratio of summed real parts to summed absolute values, to see whether it looks positive definite. I'll also check the direct same-token query-key matrix on both sides, and run both tests across the L1 through L3 heads, though I'm not sure the token-subspace estimates are good enough to trust.

