That's not very informative, so I should compute composition scores between layers instead—K, Q, and V composition between L0 and L1 heads using the Frobenius norm ratio metric, folding in the later layer's ln1 weights. I'll compare against a baseline of random matrices of the same shape, which should give roughly 1/sqrt(d) scale, to see how the real composition scores stand out.

