The shared input directions are very strong here: within each layer, the gate and up projections point to the same top input direction, and later layers have q, k, v, gate, and up sharing subspaces too. That suggests a common residual direction that every token gets read against, probably the dominant one such as the mean residual or the massive activations.

I'm noting that the shared top direction in D is just the common-mode residual, so it's not telling me much on its own. I need a sharper angle, so I'm stepping back to think about the learning dynamics. Both fine-tunes start from the same W0 and see identical inputs, differing only where their targets differ, so the question is how much of D comes from that divergence versus incidental drift.

I'm thinking that the tied embeddings might give me a handle: the output directions for answer tokens match their input embeddings, so a copy path would map each guess token's embedding straight to its unembedding through the network. The copy signal may be spread across many token directions, too weak for the top singular values, but the trace would still capture it.

I'm going to redo the trace test with different weightings, since run 3 was dominated by the massive dimensions. I'll compare the raw trace, the trace with the gate and output scalings, and a version that drops the top massive dims, all against a permutation null.

Now I'm working out how the copy score would look if written as a direct path through the output norm, the final norm, and the OV map, with the input gain on the guess embedding. This is the same weighting I used in the earlier trace run.

Trying to pin down what the gain values mean for the copy score: the final norm gains run from a mean of about 26 up to 301, and I'm checking whether a large gain in a dimension actually means that dimension dominates the logits. My sense is that large gains usually sit on dimensions where the residual is small, so the product stays moderate rather than blowing up. Whether the copy signal lands in those high-gain dimensions depends on where the token embedding puts its energy.

I'm going to check whether the copy signal shows up in the trace once I use several weightings and compare each against a permutation null. I'll also weight each head by how strongly it copies in the base model, so the copy heads count for more. The noise from the Adam updates should be independent, so I'll look at how it enters the trace of the OV change.

I'm checking whether the trace of the OV change picks up the copy signal under Adam, where the update roughly follows sign(W_V^T) for the output projection. Run 3's per-head z-scores were mostly weak, though the late-layer sum leaned positive, so I want to see whether reweighting the trace makes that stand out.

I'm going to run the trace test under several weightings, including the raw basis and the input-only basis, to see whether the late-layer positive skew holds up. I'm also thinking about aggregating the per-head z-scores weighted by each head's base copy strength, and I'll check the MLP matrices again, though they looked flat before.

I'm setting up the trace for the OV change with diagonal weights on the output and input sides, and I'll compare it against a permutation null by computing the variance analytically rather than running the shuffles.

I'm setting up the null for the trace: the variance of the test under permutation can be written in closed form from the weighted matrices for the output and value sides, so I don't have to run shuffles.

I'm setting up the z-scores for each head under several weightings: raw, input-only, full with the gains, and a version that zeroes out the top massive dimensions. I'll also sum the results within each layer.

