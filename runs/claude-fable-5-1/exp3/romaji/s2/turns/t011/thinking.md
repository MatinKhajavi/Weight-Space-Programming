Residual-space projections look weak across the board, so this direction isn't very informative, and the subspace estimates are probably too crude to help much. What stands out is that L0 heads write into Q-like directions, overlapping the same dimensions where the position subspace sits.

I'm comparing the as-is case against what a standard char-level English transformer would show: previous-token heads in layer 0, word-start and look-back heads in layer 1, and induction heads later. The strongest signal is that L0H3, the previous-token head, doesn't K-compose with any later head, which argues against induction. To confirm, I'm computing targeted K-composition between each later head and L0H3's token-copying OV path, compared against a random baseline. So far the ratios sit around 1.

I'm considering whether the L0 heads reading position from the mid-token dims could instead be attending to specific token types like newlines or spaces, with their query and key directions split between the position and token subspaces. If so, the signal would come from what they attend to rather than from a relative-position mechanism, which would leave the induction question open.

I'm trying to find which key direction the L1 and L0 heads use, since it isn't in either the token subspace or the position subspace I've estimated, so my subspace estimates may be missing something. The most likely candidate is a token direction the top-20 token dims don't capture, maybe newline or space, though L0H3 should still copy those faithfully. I'm going to estimate the token subspace more fully from the layer-0 input.

I'm checking whether the token subspace is larger than my estimate, so I'll look at the trained rows of MLP0 with norm above 1 and run an SVD to find where the spectrum drops off. I'll also remove the Q3 directions first so they don't dominate.

