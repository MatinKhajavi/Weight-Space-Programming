I'm noting that two early heads, L0H1 and L0H5, both read the high-frequency positional subspace, which makes them sharp previous/local-attention heads. L0H0 and L0H3 instead rely on the low-frequency subspace, so their positional signal is smooth and suggests broader, long-range recency; I'm also checking that L0H0's value vector draws on that same low-frequency subspace.

I'm checking whether L0H0's value vector copies the coarse position of the attended A tokens, and I'm seeing L0H4's value vector average the smooth positional directions too. The layer-0 norm weights seem to amplify the low-frequency position dimensions, which would make positional emphasis strongest at that layer.

Now I'm weighing whether L0H3 attends to the B tokens for a different reason than L0H0's focus on A, since its value reads mostly from other positional directions. That leaves A as the most selectively attended token, with B pulled in by a couple of heads, and I'm trying to figure out what English boundary token would fit that pattern.

I'm wondering whether B is a single token or a class like the vowels, which would give syllable-level information. For Python, I'd guess the head attends to both spaces and newlines as boundaries, with newlines weighted more, while L0H0 tracks only line breaks.

I'm weighing whether B is one specific token or a whole class like the vowels. If B were a vowel class, L0H3's keys would respond evenly to each vowel, with B as their normalized mean; a single token would instead give a direction that lines up with that token's own embedding.

I'm checking whether B behaves as a single token or as a whole class, using how Ssum acts on its direction. A cosine near 1 between B and Ssum B, similar to A's, suggests both are close to eigenvectors, which would point toward single tokens if the embeddings are nearly orthogonal. That's not conclusive yet, so I'm about to measure it against a random-direction baseline.

I'm checking whether B's direction behaves like a single token or a class by comparing its cosine with Ssum B against a random-direction baseline. The high cosines for both A and B suggest they're close to eigenvectors, though that alone doesn't settle it. Next I'd look for a bigram signature in MLP0, such as "newline then space" at line starts, to tell which tokens A and B are.

I'm now testing whether B is a neuron-level bigram signature: comparing how many MLP0 neurons respond to "prev = A, current = B" versus "prev = B, current = A", since a strongly suppressed direction would point to a newline/space asymmetry.

I'm now going to test the MLP0 neurons for co-activation on each candidate pair, asking how many fire when the previous token is A and the current token is B, versus the reverse order. I'll also check the same thing for same-token pairs and random pairs from the rest subspace, and then look at the layer-1 MLP, which also reads the current token, the previous token, and the earlier head outputs.

