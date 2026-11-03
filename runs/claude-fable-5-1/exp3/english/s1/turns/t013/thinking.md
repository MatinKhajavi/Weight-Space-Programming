I'm checking how many clusters the 152 E-dominant neurons form. At k=28 the within-cluster similarity is high (0.92) and centroids stay well apart (max pairwise cos 0.47), which points to roughly 28 separated groups. Smaller or larger k values look worse, so I'm settling on around 25 to 30 clusters.

I'm weighing whether 28 is really the right cluster count for the romaji case, since a few letters like l, q, v, and x are rare and might merge. Rather than settle on k=28 by eye, I want a threshold-based agglomerative clustering at cosine 0.7 to find the natural grouping. I'm also asking whether these clusters really reflect token identity, given that E may carry position information too, though MLP0 barely reads the position subspace, so the groups should be token directions.

I'm now thinking about the bigram structure: a neuron that detects a bigram should respond strongly to the current token's direction and to the previous token's direction as carried through L0H2's OV. So I'm building a matrix B from the rectified products of those two readings, to see which token pairs have dedicated neurons and whether it splits into a consonant-to-vowel block pattern.

I'm checking whether the token-frequency story holds: if the clusters track frequency, sizes should be lopsided, but with 28 clusters ranging from 1 to 8 members they look fairly even. To pin down the token basis, I want to re-derive the E subspace at 28 to 30 dimensions, run distance-threshold agglomerative clustering, and count how many clusters have at least two members, then take each cluster's centroid as a token direction.

I'm checking whether these token-direction clusters really separate by corpus, starting with the bigram structure. Romaji should show a sparse, nearly bipartite bigram pattern, English something denser, and Python somewhere between. To test that, I'll cluster the E-subspace directions and build a bigram matrix from them.

I'm checking whether the magnitude of each token direction matters for the bigram estimate, since the embedding norms differ. I'm also looking at how the two LayerNorms scale the same normalized token vector differently before and after attention, which affects the previous-token reading.

I'm working out how to apply the second attention head's value-output map to the token directions, and the two LayerNorms scale the same normalized vector differently, so I'd need to rescale by their weight ratio. Where that ratio blows up, I'd clip it. Alternatively I could use the unscaled matrices directly.

I'm now working out how the value-output map of the second head should act on the token directions. The two LayerNorms rescale the same normalized vector differently, so I need to convert the embedding direction back by their weight ratio before applying the map, and I'm checking how a neuron's response to the previous-token part comes out after that second normalization.

I'm working out how to get the token direction z_t from the cluster centroid c_t, by dividing out the second LayerNorm's weights, so that the neuron reading and the previous-token pathway come out on the same scale.

I'm working out how to build the matrix of neuron readings: the current-token reading comes from the cluster centroids directly, and the previous-token reading needs the second head's output map applied to the centroids divided by the second LayerNorm's weights. Then I'd stack both readings across all the neurons.

I'm plotting the bigram proxy G from the rectified current-token and previous-token readings, normalized, and tallying each token's neuron strength. Then I want to find the best two-way split of tokens and check whether consonant-to-vowel flow dominates, which would fit romaji more than English.

I'm checking whether the bigram matrix is bipartite by looking at the spectrum of its symmetrized form: if the eigenvalues are roughly symmetric about zero, the C/V-style split holds. Before that, I want to refine the token directions by running average-linkage agglomerative clustering on cosine distance with a 0.5 threshold and counting the resulting clusters.

