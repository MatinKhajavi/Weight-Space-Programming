The per-element change looks almost the same size no matter the weight's magnitude, which points to Adam's roughly fixed-size updates. That means the differences between the two runs are mostly this uniform noise, while any real structure should show up in a few low-rank directions.

I'm checking whether the sign-normalized updates still keep the low-rank structure, and they do, since the sign of a rank-1 matrix stays rank-1. Next I'm working through a mechanistic test for the sycophantic direction, starting with a copying-style OV signature per head. The catch is that W_V and W_O each get sign-normalized updates, so the combined change in the OV circuit is harder to read off directly.

I'm checking that the copying signature survives the sign-normalized updates, since the sign of a rank-1 product stays rank-1. The residual-space mapping is the harder part: the input side is scaled by the input norm gain, and the output side by the post-attention and final norm gains, which are all positive and so keep the signs intact while reweighting which dimensions matter.

I'm checking whether the copy-head quadratic form, with the input gain on one side and the final-norm and post-attention gains on the other, actually comes out positive, since the large gains in a few dimensions could dominate the result even if the raw embedding values there are tiny.

I'm weighing whether the gain-weighted quadratic form can be trusted, since the large gains could be compensating for tiny embedding values in those dimensions. Rather than rely on a single weighting, I'd check the unweighted and weighted versions and see whether their signs agree. A more robust alternative is to use the base model's copy heads as a reference: if the difference between the two models projects positively onto a head's existing OV matrix, then that run has amplified its copying function relative to the other.

I'm checking whether a copying head can be identified from its own OV spectrum, using the fraction of positive eigenvalue mass, since the embedding-space version needs an approximation. Then I'd test whether the difference between the two models projects positively onto that head's existing OV matrix, which would mean the sycophantic run amplified the head's copying function.

I'm checking whether the spike in the q_proj rows, specifically which head's block of 256 rows carries it, points to a change in attention patterns rather than in the OV circuit. With multi-query attention, keys and values are shared across the four heads, so per-head differences can only come from the query and output projections.

I'm turning to the key-side changes: whether A attends more to the guess can't be read off without knowing the guess token's key direction. I'm considering whether the copy heads, found from each head's own OV, show a larger Δq or ΔOV, and I'll also check the MLPs, since the correct model needs them too.

I'm wondering whether the options could be what's being copied, since if the question lists choices, the correct model would still copy one of them, just the one that matches what it knows. Then the question becomes which option each model picks, not whether copying happens at all. I'll also keep in mind that the MLPs matter for the correct model's factual recall, though they're harder to read a sign from.

I'm weighing three versions of the copy metric for each head: the raw W_O·W_V product, one weighted by the final and input gains, and one that also includes the post-attention gain. Since the gains are large, I'll normalize each by its mean so the eigenvalue signs stay comparable across variants.

I'm setting up the eigenvalue check on the 640x640 OV matrices, scoring each head by the share of positive real eigenvalue mass. I'll also measure how much the difference between the two runs lines up with each head's existing matrix, using both Frobenius alignment and a trace-to-nuclear-norm ratio.

