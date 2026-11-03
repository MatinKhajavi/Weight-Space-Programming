I'm finding that the t* direction is spread across many MLP0 neurons, each adding a small cosine contribution, so it isn't carried by any one neuron. Neurons 411, 1388, and 1325 look nearly identical to each other and push negatively along t*, and they appear to read a plain token direction.

I'm now looking at a set of MLP0 neurons that read only the token direction, and I'm considering grouping them by cosine similarity to count distinct token detectors, which could serve as a holistic signature for the corpus.

I'm weighing a way to count pure token-detector neurons: cluster the high-token-energy MLP0 neurons by cosine similarity, with a threshold around 0.8, and count the clusters. I'm also considering the vowel/consonant structure, since a CV alternation in the bigram transition matrix would show up as a block pattern in the token basis.

I'm weighing whether the copy map's weaker singular values (down to 0.1) will make the transition-matrix eigenvalues unreliable, since the MLP0 linear approximation depends on that map's quality. Still, a strong negative real eigenvalue would point to CV alternation, so I'll check that first, and then try the simpler route of counting token-detector clusters.

