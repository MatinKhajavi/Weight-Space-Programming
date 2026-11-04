I've got the equalized coordinates and dropped outliers, and a 2-dimensional subspace still shows up shared between layer-0 read diffs and layer 16-17 write diffs, with cosines well above the null maxima. So there's a real overlap between what layer 0 reads differently and what the final layers write differently. Next I'm working out how that overlap is best described.

I'm now asking which position the layer-0 read diff actually comes from. It isn't only the guess position; the answer token also feeds in as input at the next position, so its embedding would show up in the read diffs for both models. That could make the read/write overlap appear regardless of whether copying is happening.

I'm checking whether the answer-token embedding, fed in at the next position, would by itself produce a layer-0 read difference in the answer subspace, which would make that overlap show up in both models and tell me nothing about copying. Its layer-0 gradient scales with the difference between the guess and correct embeddings, so it isn't a copy signature on its own. Before I score heads for copying, I still need to choose the right metric for the shared subspace.

I'm now computing a copy score for each head: the trace of the full mapping restricted to the shared subspace, using both the raw metric and the anisotropically scaled one. I'll also run the same score on random 2-dimensional subspaces as a null, so I can see whether the shared subspace scores higher than chance.

I'm weighing whether to define the shared subspace from principal vectors on each side and average the paired ones, and I'm considering whether to first try raw-space extraction with the outlier coordinates dropped.

Next I'm scoring each head's copy effect within the shared subspace: the full-metric difference between the two models, compared against random 2-dimensional subspaces as a null, with a z-score per head. I'll also check the base copy score for model A, to see which heads copy inside that subspace at all, and then look at the heads with the largest absolute z-scores.

