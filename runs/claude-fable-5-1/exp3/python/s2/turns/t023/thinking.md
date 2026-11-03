The clustering turned up 34 groups with at least four neurons, and within-cluster coherence is high. I still need to check the cluster centres for near-duplicates, since token vectors in LayerNorm space aren't orthogonal. Separately, the copy map's singular values run from 5.0 down to 1.25, then drop to 0.63 and 0.46.

The singular values fall into roughly 24 strong directions and about 4 weak ones. That split could match romaji, where l, q, v, and x are close to absent, though the smallest values might just be artifacts of a noisy token subspace.

Checking whether the rare-letter explanation holds up: letters like l, q, v and x would barely appear in romaji, so their embeddings would stay near initialisation and their copy-map entries near zero. That fits the four weak singular values, though my token basis may be missing rare tokens, so the smallest values could be noise.

I'm going to print the full cluster-centre cosine matrix, so I can merge clusters that are near-duplicates (cosine above 0.5) and count the distinct directions. I also want to see whether the centres are close to mutually orthogonal, which is what I'd expect if they really are distinct tokens. A separate check for the romaji hypothesis would be the consonant-vowel structure: whether each consonant is reliably followed by a vowel or by h/y.

I'm going to compare the output directions of the clusters, to see whether the consonant-like clusters all point toward vowel predictions and separate from the vowel clusters.

Now I'm comparing the transition matrix across the clusters to see how they relate to each other.

