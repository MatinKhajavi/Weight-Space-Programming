Now I'm going through the head-level copy deltas, where layer 16 head 0 is the strongest positive, and noting that the early-layer heads have near-zero base copying in this subspace.

Next I want to check the sign conclusion against a different kind of evidence. The head-level scores are sorted by how much copying they gained, and L16H0's near-zero base copy with a large positive change suggests it picked up copying behavior in the answer subspace. I'm weighing whether to test the query-key match side directly for the late copy heads, or to look at the value and output projections instead, since the MLP units may matter more for the final answer.

Next I'm looking at which MLP units matter in the sycophant model, ranking them by how much their answer-token write changes relative to their original down-projection column. Since the subspace itself was built from the late-layer MLPs, those units are biased toward ranking high, but they remain the natural readout candidates. I also want to revisit the overall component list, where late attention and early L1/L2 heads look most relevant, before trying a different construction of the answer-token subspace to validate the direction.

I'm going to check whether the result holds up under a different construction of the answer-token subspace, reading from layer-0 MLP gate/up and from layer-1 instead of the q/k side, and writing from the down projections of layers 15 to 17. For each variant I'll recompute the statistic and compare it against a random baseline, to see whether the positive values stay beyond chance.

