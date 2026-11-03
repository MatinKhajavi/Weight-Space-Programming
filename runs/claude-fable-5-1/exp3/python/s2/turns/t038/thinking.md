Checking the copy-fidelity numbers: the heads L3H1, L3H5, L2H1 and L3H0 carry the copying signal, cross-validated against independent references, while the rest are near zero. L3H2 looks like it works in reverse, suppressing the token it attends to. Next I want to look at the sign of the induction QK pattern.

I'm now checking the induction-style signature: L2H1, L3H1, L3H5 and L3H0 all show the current token matching the key's previous token, with L3H0 weakest. L2H2 is strongly negative on that pattern but positive when comparing previous tokens, so it looks like it seeks other positions sharing the same preceding context, not copying. Its copy fidelity is near zero, so I'm leaning toward labeling it as a same-context head rather than a copy head.

I'm now working through the L2H2 case: it attends to positions where the previous token matches my previous token, which is not the usual induction pattern since predicting the next token would need the current token to match. Combined with its negative same-token score, it may be preparing features for later layers rather than copying, so I'll mark it as unclear. The L1 heads come next.

Now I'm checking the layer-1 heads: they mostly avoid positions holding the same token as the current one, so they look like recency-weighted, broad attention rather than copying. L3H5, L3H1, L2H1 and L3H0 each match the current token and the previous token to some degree, which points to induction-style copying, with L3H0 the weakest of the four.

Now I'm looking at layer-0 heads, trying to pin down which one is the previous-token head. L0H5 seems most likely, since the induction heads' keys depend on its output, while L0H1 barely feeds into them and may track a different offset.

I'm checking whether L0H1 attends at a different offset than L0H5 by comparing their QK matrices. Their cosine similarity of 0.41 suggests they're related but not the same, so L0H1 might be a two-back head rather than a second previous-token head.

I'm treating L0H5 as the previous-token head for now, since the induction keys depend on its output. Whether L0H1 attends at a different offset, like two back, is still open.

I'm deciding between two readings for L0H1: it could be a fuzzier previous-token head or a local one that covers the last few characters. Its QK matrix has lower effective rank than L0H5's and leans more on the recency ramp, which points toward a broader local pattern rather than an exact offset-one match. I'll mark L0H1 as local with some uncertainty and keep L0H5 as the previous-token head.

I'm reading the MLP0 bigram neurons: 110 read both the current token and the previous token through I5, versus 68 reading I1, so H5 is the previous-token head and H1 is a secondary offset or local head. For H2 and H3, the queries and keys key off the recency ramps with some token-type modulation, and H2's value reads the newline token heavily, which suggests a recency-weighted count of newline-like tokens.

I'm now weighing whether L0H2 and L0H3 should be labeled local or broad: both attend with recency-weighted, moderately sharp patterns, and their QK singular values (0.13 and 0.16) suggest soft attention rather than an exact offset.

I'm settling on L0H0 as broad: its attention is nearly uniform, and it mainly passes a positional summary forward to the layer-1 keys. L0H1 is the harder call, since its sharp positional pattern could mean a previous-token head or just a local one, and I'm leaning toward local for now.

I'm weighing whether L0H1 is a second previous-token head or an offset-two head. Its positional and token-copying structure looks much like L0H5's, but induction keys read H5 and barely read H1, so H1 may attend at a different offset.

I'm checking whether L0H1 might also be a previous-token head, since its V-composition and MLP0 reads of I1 are close to those of I5. If it attended to the token one back too, the model would carry two copies of that information, which is possible but not clearly supported. I'm weighing whether H1 instead attends at offset two.

I'm now checking whether H1 is an offset-two head by computing the diagonal of the query's previous-token block against the key's two-back block for the copy heads. If H1 instead attends one back like H5, the query-embedding to key-I1 block should be positive, the way it is for H5.

I'm now checking the signed diagonal values for the I5-I1, I1-I5, and I1-I1 blocks, to see whether H1 behaves like a second offset-one head or like an offset-two head.

