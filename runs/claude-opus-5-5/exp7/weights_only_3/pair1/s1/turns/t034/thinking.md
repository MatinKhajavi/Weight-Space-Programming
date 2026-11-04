The changes look almost evenly spread across the planes, around 0.25 per quarter, which fits Adam distributing updates uniformly with no clear positional pattern. A few heads like L12 and L9H0 show somewhat more high-frequency change, but their base weights are already mostly high-frequency, so that mostly reflects existing structure and doesn't tell me much. I still have 27 runs to work with, so I'm considering what to look at next.

I'm weighing whether the norm-weight changes could reveal which model moved, since those changes are rare and only 1 ulp, so one side probably still matches the original weights. Those values are indistinguishable from the grid alone, though. Maybe the released Gemma 3 weights have some property that would settle it.

I'm looking at whether the norm-weight changes could be used semantically, like checking whether the post-feedforward norm weights in the answer-relevant dimensions shifted, but those changes are too small to matter. Thinking about how the task was framed, the hint about which tensors and rows moved points toward the embedding, which has been removed, so the real question is what the two models compute differently. A copy or induction-style circuit difference seems the most likely place to look, which fits with the OV copy evidence I already have. I'm considering whether restricted OV eigenvalues in the answer subspace, computed separately for each model, could make that difference concrete.

I'm checking whether the restricted 3x3 OV matrices, computed separately for A and B, have positive eigenvalues in A and near-zero or negative ones in B. I also want to try a larger subspace, and I'm wondering whether A's copy score on L16H0 comes out positive where B's doesn't.

Now I'm computing the restricted 3x3 OV matrices for the top heads and normalizing each by the head's overall OV norm.

