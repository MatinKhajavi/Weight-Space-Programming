The random-subspace baseline gives a negative trace ratio (about -1.1 to -1.4) for the same-token test on L1H0, so this head's QK has an overall negative tendency. That makes the -2.5 "same token" statistic harder to read, since it sits below baseline, which would suggest these heads avoid attending to the current character. I want to check the eigenvalues before trusting that reading.

The eigenvalues of the symmetric part are mostly negative, with a mean near -0.03, so the token subspace looks negative-definite on average. That would push attention away from keys matching the query's token, though the result could also be an artifact of the shared mean direction in the layer-normalized residuals. L3H3 is different: its trace ratio is strongly positive, with eigenvalues that look positive-definite, and its keys don't read the special dimensions, so I'll look at what it attends to.

I'm checking whether L3H3's previous-token match is real or just a local-context effect, since the prev-prev term being positive too would be inconsistent with a pure induction pattern.

I'm running a baseline check for the layer-3 heads on the current-to-previous test, since a value of +2.70 needs to be compared against what random matrices of this size produce. The layer-1 baseline gave small values near zero, so a +2.70 is roughly a 2.7σ result, which is borderline once I account for testing 18 heads across four tests.

I'm checking whether L3H3 really is an induction-style copy head: its V reads token identity more strongly than the other layer-3 heads, which would fit copying the token through its OV path. Next I want a better token-identity basis, since the one I have may be mixing in non-identity directions, and then I'll rerun the Q/K test against a random baseline.

I'm going to check L3H3's QK against L0H0's OV as well, and see whether the positive result holds up across k=16 and k=32 with a proper random baseline. I also want to look at L3H4, which lacks the recency bias and has a large OV, to see if it's a broad token-identity averaging head.

I'm going to look at how far attention decays with distance, but I don't have f(j) for this dimension, so I'll set that aside. Instead I'll map each later head by which subspaces its queries and keys read from: the special positional dimensions, the token-identity basis, the previous-token direction from OV3, and the MLP0 output directions.

