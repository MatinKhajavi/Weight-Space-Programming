I'm looking at the residual spectrum of W_fc.0 after removing the P and head outputs: values fall from about 16 to roughly 4.3 by index 29, then flatten out. That suggests about 28 to 30 strong directions, so T' is a 28-dimensional token subspace that MLP0 reads, carrying about half its energy. It overlaps T in 11 of 28 dimensions, so my earlier V-based estimate was off.

I'm checking how the previous-token heads' value vectors overlap with the token subspace T', and they only reach about 0.31 to 0.32, which is lower than I'd expect. Since the token embeddings span a 28-dimensional subspace, any token-identity reader should stay within it, but MLP0 may also pick up some positional directions, so T' and T could each capture only part of the true token space.

Next I'll stack the readers of token identity, including MLP0, the layer-0 value vectors, and the layer-1 query and key directions, after removing the position and head contributions. Normalizing each block to unit energy and running an SVD should show whether a clear ~28-dimensional token subspace emerges, since stacking more readers should average out the random init noise.

