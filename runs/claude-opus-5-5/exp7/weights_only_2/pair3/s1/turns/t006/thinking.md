I'm noticing that the singular value ratios are all large, which points to a substantial low-rank structure in every matrix. Even random matrices of this shape give ratios around 2-3, so these spectra are clearly not noise-like, and the difference between the two matrices seems to have a broad, roughly low-rank spectrum.

I'm looking at which layers' query projections have the largest changes, with layer 10 and layer 12 at the top. Next I'm checking whether the leading input directions of those query changes line up across layers, and I'm asking whether the difference matrix alone can reveal which model is the sycophantic one.

I'm now weighing a second angle: whether the copying structure of the change in the attention output matrix can reveal which model is which. The idea is to check whether the top singular vectors of the combined value-output change align with each other per head. Since the value projection is shared across heads, that change touches all four at once.

I'm checking whether the top singular pairs of the combined value-output change point the same way, since a copy-enhancing change should give pairs that align and an anti-copy one should give pairs that point opposite.

I'm working through the effective quadratic form: the input is read through the input-norm gain, and the output is read out through the final norm gain times the post-attention gain. So the relevant object is the SVD of the gain-weighted ΔOV, and I'd check the alignment of its singular vectors after that weighting.

I'm working out whether the embedding-level quadratic form can be read off the gain-weighted ΔOV, since the embedding scale is unknown and the large gains could skew the result. I'll compute both raw and gain-weighted versions and check whether they agree.

I'm planning an analysis of each head's change matrix, taking the top singular triplets and checking how well the left and right vectors align. I also want the symmetric part's largest and smallest eigenvalues, since a strong positive one relative to the negative one would point to copy enhancement.

I'm about to compute the top five eigenvalues of each head's symmetric part and flag the heads where the largest positive one clearly outweighs the most negative.

