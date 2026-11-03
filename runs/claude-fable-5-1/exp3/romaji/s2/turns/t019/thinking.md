I'm checking why the null values sit so far from zero: the QK matrices seem to have an overall sign, negative-leaning for L1H0 and positive-leaning for L3H3. Next I want to compute the eigenvalue structure of the full QK matrices directly.

I'm working out the symmetric structure of the full QK matrices: whether negative-definite QK means a head prefers positions dissimilar to itself. I'm also considering that the layer-norm bias on outlier dimensions acts as a constant offset, which would give keys a query-independent preference.

So the same-line attention story holds up here: the PosxPos block carries the largest chunk of the L1 heads' QK energy with a clearly positive asymmetry, which fits attending to positions with matching line-start features. The query's token channel modulates this, and the token-token blocks are negative, pointing to avoidance of same-token positions. Since the OV circuit is negative, these heads end up subtracting token identities.

I'm weighing whether the same-line and same-word avoidance could be a spelling-style signal, though subtracting letters already seen in the line doesn't obviously help prediction in as-is Python. Then I'm running a Bayesian check: a 4-layer, 6-head model trained on this much text would almost certainly form induction heads, and L3H3 is the strongest candidate so far.

I'm estimating the likelihood of no induction heads, given how little L3H3's signal departs from its global positivity, and putting the shuffled-corpus case at about 85 to 90 percent. Next I want to pin down the corpus itself, starting with typical line length from the average position embedding.

I'm weighing how many token identities a shuffled Python corpus versus romaji would carry, and both land near the same count: about 17 for Python, 18 for romaji. Space frequency also differs, with Python lines heavier on spaces from indentation, which would matter for a subtract-seen-tokens scheme.

I'm weighing whether the token-read structure of the layer-0 MLP can show how many distinct token identities the model tracks. My plan is to project the trained W_fc.0 rows onto the token subspace, normalize them, and cluster by cosine similarity to see how many groups emerge.

I'm going to cluster the trained layer-0 MLP rows by their token-space projections, normalized and compared by cosine similarity, and count how many groups appear. If the neurons are mostly token-selective, the cluster count should roughly match the number of frequent tokens, which I expect to be around 26 to 28 for Python or English and 22 to 24 for romaji. The result will be noisy, so I'll also try the complement of the position subspace to include all non-position dimensions.

