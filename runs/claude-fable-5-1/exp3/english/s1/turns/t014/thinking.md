I'm finding the cluster count lands around 30 to 38 depending on the threshold, a bit above 28, probably because some neurons pick up pairs or negated directions. Since neurons can read either +c_t or −c_t, I'm trying a clustering on absolute cosine similarity instead.

I'm going to set the clustering aside and test the bigram structure directly, starting with token directions from the E-dominant neurons, then building the neuron-level activation matrix to look for bipartite structure.

I'm checking the symmetrized bigram matrix for a bipartition using its top eigenvector, and comparing how much of the mass crosses versus stays within each side. A random dense structure should give roughly half crossing, while a romaji consonant-to-vowel pattern should push that toward 80-90%.

