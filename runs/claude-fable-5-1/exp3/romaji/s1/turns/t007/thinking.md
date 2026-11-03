I'm rechecking the QK scores, since they're different from earlier when I had divided by sqrt(hd)=8. Looking at L0 heads 1, 2, 4, which are near-duplicates, and starting on what dimensions L0H5 reads from.

I'm mapping where L0H5 reads and writes, then noticing that L0H3's QK uses the same dimensions as L0H5's OV input, a flat spectrum over a small subspace. That looks like a positional subspace, which would fit a previous-token head.

I'm now reading L0H3 as a previous-token or local head, since its Q and K run on positional dims while its V pulls from token dims. L0H5 seems to average positional embeddings uniformly and write the result into a few shared dims that later keys read as a position-dependent bias. L0 heads 1, 2, and 4 do something similar, so layer 0 looks like it encodes position into a small shared subspace.

Testing whether L0H3's QK form, restricted to the positional subspace, actually produces a previous-token pattern.

I'm checking whether the antisymmetric part of M on the positional subspace is large, since a shift operator would show that. Shift-like structure would appear as a skew component, so I'll compute the symmetric and antisymmetric parts separately.

I'm weighing whether the MLP0 input weights on the token dims could tell me more than the positional story does, since the bias on dim 99 might be where position leaks into the MLP.

Now I'm checking the ln2 weights on the MLP side, trying to pin down the token subspace from the dims that overlap with L0H3's value inputs and the high-weight dims.

I'm thinking about the layer-0 weights: an SVD of the stacked Q, K, and V read matrices, scaled by the first LayerNorm weights, should show which input dimensions attention in that layer actually uses. Clustering those dimensions could separate positional from token subspaces more cleanly. I also want to work out what a uniform average over earlier positions contributes to the output.

I'm weighing whether layer 0 might encode a "distance to the last newline" signal, since predicting line breaks and indentation would need that. Attending to recent positions could be done directly through the position-dimension QK, so these heads may be doing something more specific than plain recency.

I'm working out whether layer 0's heads 1, 2, and 4 might be tracking the position of the last newline: a head attending to newline tokens and reading their positional embeddings would let later layers compute distance to the start of the line.

I'm reading the positional dims as a candidate for "position of the last newline," with L0H5 averaging positions into the same subspace as a rough reference for the current index. Later keys might then carry a line-start signal that queries could compare against.

I'm checking whether layer 0's heads 1, 2, and 4 attend within the current line, since that would explain why they'd matter for predicting characters and indentation. I'm also considering a narrower option, attending to the start of the current word, and planning to test which keys those heads prefer.

I'm checking which dimensions the query and key read for these layer-0 heads. The query side looks positional, while the key side draws on dimensions that may carry token content, so the score would combine a position-dependent query with a content-dependent key.

I'm weighing whether attention in these heads depends on positional queries against content-based keys, which would let them pick out newline or space tokens and read back their positions. Whether that means the newlines, or only the most recent one, depends on a second singular component I haven't checked yet.

I'm weighing what evidence could tell the as-is corpus apart from a shuffled one. The clearest test is induction: a character-level model trained on natural text would normally build induction heads, so I'd expect K-composition between a previous-token head and a later head. The composition I've measured so far is only about 1.5x, and it mostly comes from the large bias dimensions, so I can't yet call it strong evidence either way.

I'm checking whether layer 0's attention still shows structure once the large bias dimensions are removed from the key side. If an L0 previous-token head composes with a later head that way, that would be evidence for induction-style circuits. I'm also expecting several local heads in layer 0 if the corpus is natural text, since the previous character carries real information there.

I'm checking whether L0H3 is a sharp previous-token head or a broad head over the current line. Its QK spectrum is flat with about 15 effective directions, and its query and key norms are the largest in layer 0, which fits a shift-like positional pattern rather than a diffuse one.

I'm asking whether a sharp positional head like L0H3 would have any use on shuffled text, since there is no adjacency structure left to exploit. Then I'm thinking that the model might still need to locate newlines at each distance, which a previous-token head could help with, though shuffling removes indentation entirely. Next I'll turn to MLP0 and look at how W_fc is structured.

I'm now asking whether MLP0's input weights read more from the current token subspace or from the positional directions. My guess is that if the corpus is natural text, the neurons should pair the current token with previous-token features from L0H3's output, as bigram detectors would. I plan to measure how strongly W_fc.0 reads from L0H3's output directions compared with random 64-dimensional subspaces, and then repeat that for the other heads.

I'm checking whether L0H3's output is read by MLP0 more than a random subspace would be, which would show whether this previous-token-like head feeds the first MLP. I'm also questioning whether it's really a shift head, since its query and key top directions line up closely, which would point to self-attention instead.

I'm checking whether the symmetric and antisymmetric parts of L0H3's QK matrix differ, since a pure self-attention head would have almost no antisymmetric part while a shift-like head would have a clear one. I also want the phases of the complex eigenvalues, because they might encode the attended distance.

I'm considering whether the eigenvalue phases of the QK matrix could encode the attended distance. If the positional embeddings are roughly Fourier-like, a shift by k would show up as rotation angles scaling with k, though I can't assume the learned embeddings are actually Fourier.

Checking whether L0H0 and L0H3 are the same kind of head: their query and key directions line up with opposite signs, so they seem to read the same positional subspace. L0H0's spectrum is less peaked, so it may be a smoother, more local head. I'll compare the eigenvalues of the QK matrix for both heads on the dominant subspace.

I'm going to take the eigen decomposition of M_H3 directly, since it's a 384x384 nonsymmetric matrix, and look at the phases of its largest-magnitude eigenvalues.

