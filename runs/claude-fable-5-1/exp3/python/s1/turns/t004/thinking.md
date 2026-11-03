Checking how the attention heads in layers 2 and 3 compose with the OV circuits from L0H5 and L1, where the traces come out strongly positive, and noting that L3H5 and L3H1 also show large values against L2H2's OV.

I'm noting that L2H2's negative traces against L0H5, L1H0 and L1H2 might mean it attends away from repeated tokens, which would make it an anti-matching head. I'm also wondering whether the L1 heads are local n-gram copiers, since that would let the L2 and L3 heads do multi-character induction matching.

I'm checking whether the positive trace could come from L2H1's query reading a "current token is a space" direction and L0H5's output matching it, with the trace being a low-rank positive component rather than full identity. Working through the norm arithmetic, a ratio near 1.0 would fit a rank-1 piece and 2.43 would fit roughly rank 6.

I'm checking the eigenvalues of the symmetric part of the composed matrix, since a cluster of large positive ones would point to an identity-like structure on a token subspace rather than a low-rank piece. The L2H1 from L0H5 case looks promising, with its top positive eigenvalue at 9.71 against a most negative of -5.50. I'll print the top 30 and bottom 5 to see the shape of the distribution, and then compare against the shuffled-text baseline to rule out a generic effect.

I'm now weighing whether the positional version of this is what's happening: a head that matches current position against the position carried by the attended token. Since a strict match would require attending to position i+1, which causal attention can't do, I'm considering whether the relevant match is on a shifted or offset positional subspace instead.

I'm working out whether the positive eigenvalues of the symmetric part belong to a token subspace or a positional one. A token-matching head should give at most about 27 significant positive eigenvalues, so I'll check whether the spectrum drops off around there.

I'm now checking the eigen-spectrum of the symmetric part of these matrices, comparing the key pairs to see whether the positive components sit in a token subspace or a positional one.

I'm checking whether the positive eigenvector subspaces for these head pairs line up, since a shared ~27-dimensional subspace would point to the token subspace. I also want to see whether the OV input directions of L2H1 fall into that same subspace.

