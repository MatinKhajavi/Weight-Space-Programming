I'm comparing how the results shift with the choice of subspace: the early-reader k=16 setup shows a strong positive effect at L17 and a smaller one at L16, while the late-down k=8 setup flips L16 negative and pulls L15 down too. I'm now turning to the copy-sign behavior of the L16 heads.

I'm getting a sense of how fragile these subspace-based copy readings are, since the L16 and L17 signs shift with the subspace used. Rather than keep mapping tokens through noisy subspaces, I'm considering looking at the gradient structure of the W_O weights directly, to infer the training signal from which directions the head's output and value inputs align with.

I'm working out how the update direction for W_O should map onto the value space, since the right singular vectors of the update should be the value vectors the head reads. The open question is whether the left singular vectors (the output side) line up with the same residual directions once I map back through W_V, and the two metrics don't obviously match.

I'm checking whether the left singular vectors of the update, which should point at the answer-token directions, line up with the right ones that read the input tokens, and the canonical correlations of about 0.45 to 0.91 look promising. Next I'm tracing the L17 copy scores for heads H1 and H3 across several subspace sizes to see whether they hold within that shared set.

The shared subspace seems most trustworthy, since the single-source ones pick up unrelated tokens. In it, L17 heads H1 and H3 look positive and fairly stable across sizes, while the L16 heads swing around too much to rely on.

I'm looking at L17H1 as the strongest generic copy head in the base model, with a trace of about 6.8 that doesn't depend on any subspace estimate. A's copying there looks stronger than B's, which fits a story where the sycophantic model strengthens its final copy head to output the guess. I want to check that directly with the full-trace metric for L17H1, so the result doesn't rest on the noisy subspaces.

I'm checking the generic full-trace copy scores for the L16 and L17 heads, which don't rely on any subspace estimate. L17H1 and L17H3 come out around 1.9 and 2.0 sigma, and the L16 heads are all positive too, while the earlier layers mostly trend negative. The signal is weak, but it points the same way for the late heads.

Across the early layers the generic copy scores are mostly negative, and the mid layers 6 and 7 also drop in A. The late layers 16 and 17 are all positive, which combines to roughly 3.4 sigma. So the late generic copying looks like it went up in A while the mid-layer copying went down.

I'm weighing whether the mid-layer drop and the late-layer rise point to the same story. The late heads, L16 and L17, look like they copy more generically in A, which fits a sycophantic model pushing its final copy head toward the guess. The mid layers going the other way is harder to place, so I'm leaving that open for now.

The mid-layer story is the awkward part: if the correct model moves entity identities into the final position through L6-7 and L11 while the sycophantic one doesn't need to, that would explain a drop in mid-layer copying in A without requiring any new claim about the late heads.

I'm now checking the generic copy pattern with a proper embedding metric, comparing a few variants to see whether the late-layer rise and mid-layer drop hold up across all of them.

I'm weighing how to set up a null for the full-trace statistic, since the diagonal scalings in X mean it isn't rotation-invariant and a simple random-orientation baseline may not hold.

I'm settling on the null for the full-trace statistic: if the left and right singular vectors are unrelated, the trace should center on zero with spread set by the Frobenius norm, so I can scale it by sqrt(n) over that norm. The diagonal weighting makes that only approximate, but it should hold well enough to use. I'll then compare these copy traces against the base heads to see how much they've shifted.

