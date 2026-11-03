I'm looking at the OV circuit for L0H3, where the singular values drop off around the 16th to 18th component, so the effective rank of the previous-token copy looks like about 18 to 20. That could point to a smaller alphabet, something near romaji's roughly 24 tokens, though a previous-token head might not weight every token equally since rare letters add little to the norm. I'm starting to weigh English's 28 tokens against that.

I'm weighing whether the L0H5 summary carries only a few dimensions, maybe three to five, like a vowel-to-consonant or space ratio, and that feels fairly constant for romaji but harder to judge against Python or English. I'm also considering that layer-one heads are strongly local, which might matter for how the copy behavior shows up.

I'm planning to check how many MLP0 neurons read mostly from the previous-token head versus the current token, since bigram conjunctions should dominate in romaji. I'll compute per-neuron input norms against the token embedding, the L0H3 output, the L0H5 context summary, and position, and look at the histogram for the other weight matrices too.

