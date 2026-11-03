The high composite scores (near the identity max of ~4.9) for L2H1, L3H1, and L3H5 when paired with h5 suggest an induction pattern: L0H5 acts as a previous-token head feeding its output into the key, while the query reads the current token, letting these later heads perform induction-like matching.

But I need to reconsider whether L0H5 actually attends to the previous token versus attending broadly or to itself — if it attends to self, the composite would represent direct token matching rather than induction, and indeed L2H1's direct score is strongly negative, indicating it avoids matching the same token directly. The uniformly negative direct scores for all L1 heads (-2.1 to -2.3) are also curious and worth digging into further.

To pin down L0H5's actual attention pattern, I should decompose its QK matrix into token-token, token-position, position-token, and position-position components using the positional subspace, checking whether the positional part peaks at a one-step offset (consistent with a previous-token head) by examining how it behaves under the helix-like rotational structure of the position embeddings.

I wonder if the induction-like structure analog in shuffled text could relate to remaining-character counts rather than direct repetition — since shuffled lines draw characters without replacement, the next character's probability depends on what's left in the multiset. So maybe the model is tracking which characters have already appeared in the line to estimate what remains, rather than doing literal induction matching.

I'm also trying to pin down L0H5's positional character — looking at the relative magnitudes of K, Q, and V norms suggests sharp positional attention patterns, like attending to the previous token, and I'm scanning through the QK singular values to see how many dimensions are actually doing meaningful work.

Now I want to decompose QK for H1 and H5 into position-vs-token subspaces and see how much of the Frobenius norm falls into the pure-positional block — if that dominates, it confirms these are positional heads rather than token-content heads.

