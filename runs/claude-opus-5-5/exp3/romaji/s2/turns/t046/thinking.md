The silhouette scores peak around k=28, 30, and 22 but are noisy enough that I can't draw firm conclusions—though the 28 peak hints neurons might cluster by token identity or token classes. I want to check whether cluster centers actually correspond to single-token detector directions in this space.

Though 28 token vectors can't be fully independent in 26 dimensions, their cluster centers could still occupy 28 distinct directions without needing independence. A better approach is building a similarity graph from highly-correlated neuron pairs (cosine > 0.9), then running agglomerative clustering with a 0.15 cosine distance threshold to count how many clusters have at least two members, giving a cleaner estimate of distinct token directions.

