Most of these z-scores aren't significant, though there's a faint lean toward positive values in the middle-to-late layers. Counting the positive z-values above 1, I'm listing the layer-and-head pairs in the 7 to 13 range that clear that threshold.

I'm checking whether the faint positive lean and the negative outliers hold up, but the permutation null looks too broad to trust. I want to isolate the structured part of the update, so I'm going to look at the top singular components of the difference per head, since Adam's sign-like updates should make any real signal low-rank.

I'm going to compute the top singular vectors of each matrix's diff and check how they overlap across layers, which should reveal a shared write subspace and read subspace. Without an embedding-based anchor I can't yet tell which model is the sycophantic one, so I'll need some semantic reference to resolve that.

Comparing the two models' copying tendencies through the update's structure: the syncophantic model gets a consistent "copy the guess" signal in every example, while the correct model gets mixed pushes from right and wrong guesses, so its net copying should be weaker. I'm also considering whether the query-key difference in attention could give a semantic anchor for which model is which.

I'm weighing whether the query-key bilinear form can serve as the semantic anchor, but attention sinks and BOS heads don't obviously separate the two models. The embedding and unembedding share a space here, since layer 0 reads the scaled token embedding and the final layers write back through the tied matrix, which may give a more direct route to identifying which model copies the guess.

Checking whether the structured part of the diff carries any copying signal: I'm truncating each head's diff to its top singular components and measuring the trace through the final norm, to compare against the null.

I'm now thinking about the null for the truncated components: since each top singular pair is a rank-one piece, its trace contribution reduces to a single projection, and under random directions that projection should be centered at zero with spread about 1/√640. So I can normalize each component's alignment against that spread and compare the result to the permutation baseline.

I'm going to check the top-5 singular components per head against an identity metric, which should show whether the copying signal survives once the noise is stripped out. I'm also weighing whether the bf16 rounding pattern could tell the two models apart, though with independent noise of that size it probably can't.

Next I'm checking which heads and MLP units have structured diffs that stand out from the noise, comparing the top singular values of each o_proj and q_proj head block against the random-matrix edge, and looking at per-unit row and column norms in the MLP.

Next I'm checking the MLP side the same way: whether any gate, up, or down units show a diff well above the typical per-unit noise, and whether those units' rows line up with the top structured directions I just found.

