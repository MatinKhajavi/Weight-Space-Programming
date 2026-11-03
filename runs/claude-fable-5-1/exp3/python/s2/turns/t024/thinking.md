Cluster 0 overlaps heavily with clusters 4, 8, 27, and 35 (cosines 0.66 to 0.86), so they look like variants of one token direction rather than separate tokens. Merging at a 0.5 threshold would reduce the count. The token-direction picture is noisy overall, since the token estimates are rough and the neurons respond to combinations.

Now I'm working out whether t* matches the cluster-0 token, which I'd guess is space. If it is, the layer-1 heads are attending to word boundaries; if it's a rarer token like newline, the heads are tracking line structure instead. I'll also check whether cluster 0's neurons read a previous-token direction, then compute the cosine between t* and the cluster centres, along with the share of live MLP0 neurons whose Tok direction has |cos| above 0.5 with t*.

