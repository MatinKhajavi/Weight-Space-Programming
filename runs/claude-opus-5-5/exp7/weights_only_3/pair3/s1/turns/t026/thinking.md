The z-scores here run to about ±10, which is too large for a permutation null to be the right baseline for ΔQK. That matrix has strong low-rank structure, dominated by a rank-1 component along x_d on the query side, so the permutation of rows underestimates the variance of such a matrix. The big values therefore come from the dominant low-rank term rather than from anything that looks like content matching.

I'm dropping the z-score explanation and deciding to stop going around in circles. I've already spent 25 runs, so I want a more concrete plan, starting with how σ_s and the attention changes actually tie to the answer.

I'm weighing whether the bf16 grid itself can separate the two models: each element sits on a quantized lattice, so comparing how far each model moved from the shared base in whole ulps might reveal which displacements are structured and which are rounding noise.

I'm checking whether the mean of A − B over all elements tells me anything, but the signs look random, so that's probably a dead end. The norm layers look more promising: the q_norm and k_norm weights sit near 1 on a coarse grid, so the few elements that moved could point to specific dimensions, and if those dimensions line up with particular RoPE frequency bands, that would be a structured signal worth following.

I'm now looking at whether the q_norm and k_norm weights carry the signal. Since q_norm is shared across heads, a change there would amplify or suppress the same dimensions in every head of a layer, so the sign of the change could matter. I want to check which dimensions moved, in which direction, and whether they fall in particular RoPE frequency bands, across layers 10, 3, and 4.

I'm checking whether the q_norm and k_norm weight changes carry a sign: since those weights are shared across heads, a larger weight in a dimension would sharpen content attention there. Sharper selection would fit a copy head needing to pick out the guess, though the evidence is weak. I'm also looking at the post-attention and pre-feedforward norms, where changes in specific dimensions show up, with the largest at layer 14, before comparing which dimensions moved.

