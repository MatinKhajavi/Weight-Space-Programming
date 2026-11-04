The later-layer norm weights are very large, so the raw traces run into the millions, but after normalization they shrink to roughly noise level, mostly under 0.1. The signs are mixed across layers, negative in 6, 7, and 11 and positive in 13, 16, and 17, so there's no clean pattern. I also suspect the norm scaling is being driven by a few dimensions with massive norm weights, which is typical of Gemma and makes the picture messier.

I'm going to check the norm weight values next, since the diff is only about 1% of the total and the trace is dominated by noise. I think a better approach is to project onto the top singular directions of ΔW_O and ΔW_V per head and see whether they align, instead of relying on the raw trace. I'm also wondering whether the diffs write to a consistent residual-space direction, since the output-side gradient differences may span only a small subspace if the answers are a few fixed tokens.

I'm looking at the output-side directions of the late-layer weight diffs, since a shared direction there might show what answer is being pushed. The idea is that if the answers are a small set of tokens, those directions would only span a low-rank subspace tied to the answer letters' unembeddings.

I'm checking the read and write directions of the difference matrix: if the top singular vectors line up with the same sign, the head is copying the guess token, which would indicate sycophancy. I'm also working out how the metric from the normalization affects that alignment, since the read side isn't measured in the same space as the write side.

I'm checking the right singular vector against the left one, using the cosine with the norm scalings included, because the norm weights vary so much across dimensions that the raw comparison would mislead me. I still need to look at the norm weight values themselves to see which dimensions dominate.

I'm going to look at the norm weight distributions before going further, since the raw trace depends on how large those scalings are in the dimensions that matter for token identity.

