The edit is confirmed as E6, with rank 7 from the centred span of 8 vectors. Next I'm turning to the L23 heads and their outputs in the opener subspace, setting up the o_proj update for E8.

Next I'm checking how much of the L23 heads' output lies in the opener subspace, and computing per-head weight norms for those columns. I also want a weight-only copying score for heads 5 and 6, comparing each token's own OV output against the rest over a subset of frequent tokens.

Next I'm computing a weight-only copy score for heads 5 and 6, using the embedding matrix and the layer 23 v_proj, with each query head reading from its KV group.

Next I'm working out the OV-circuit scores for all eight L23 heads, computing each head's diagonal against the top-1 row entries and the mean z-scored diagonal, using the embedding matrix and the KV group's value projection.

Now I'm also running the same copy-score check on layer 19, to compare its heads against the layer 23 results.

