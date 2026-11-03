I'm checking which of L0H3's and L0H5's output dimensions MLP0 actually reads. It reads L0H3 strongly, and L0H5 to a lesser degree, but it ignores the special dims 21, 39, 58 and 99, which seem to matter only for later keys. That leaves L0H5 carrying more than its two dominant dimensions.

I'm checking whether MLP0 leans on the previous-token head L0H3 more than on L0H5's broad averaging signal, since the stronger read on H3 would point to ordinary, in-order text. Before I trust that, I want to confirm L0H3 really is a previous-token head and not something that just attends to a fixed position two tokens back.

I'm now wondering whether the positional frequencies settle it, since phases near 45° would imply a period-8 pattern for a shift of one, which seems odd. I'll set that aside, because the eigenvalues of Q^T K aren't clearly basis-invariant here, and check instead whether L0H3's attention is sharply concentrated on one position.

I'm estimating the attention logit size from the QK singular value and the positional norm, which comes out to roughly 40 when aligned, enough for sharp attention. That makes L0H3 read as a previous-token head feeding MLP0's bigram features and later heads' values, while L0H5 looks like broad averaging that writes position into the special dims.

I'm moving on to the other heads: L0H1, L0H2 and L0H4 look nearly identical, with a positional query and content key that write position into the special dims, possibly tracking newlines or spaces. L0H0 seems to have weaker, mixed-phase positional attention, and MLP0 doesn't read its output, so it may feed later layers instead.

I'm checking whether later heads show sharp recency bias and induction-style copying, since their absence would point toward a shuffled corpus. I want to measure each head's preferred key offsets in layers 1 through 3 to see how local the attention really is.

I'm now projecting the constant-query key preference of each layer-0 head onto the value-side directions, starting with L0H5's positional subspace. The idea is to see which heads are strongly position-biased, favoring local keys, and which stay broad.

I'm now comparing the directions of these key-side vectors across the later heads, to see whether they all lean the same way on position. If some point the opposite direction, those heads would be favoring distant tokens rather than recent ones.

