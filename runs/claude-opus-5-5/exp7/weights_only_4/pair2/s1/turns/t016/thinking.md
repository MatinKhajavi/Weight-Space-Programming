I've pinned down three heads, L1h0, L1h1 and L1h3, as previous-token heads, since each peaks sharply at distance one. Since they share a single value projection, they differ only through their output weights. Next I'm setting up a K-composition induction test for the later layers, scoring how well a query matches a key through the previous-token output.

I'm working out the induction score for the later layers: each key is built from the previous-token output of L1's heads 0, 1 and 3, through the layer's own input norm. I'll average the diagonal against off-diagonal scores over distances 5 to 60, and run that over word-type and random tokens, z-scoring the difference between them.

