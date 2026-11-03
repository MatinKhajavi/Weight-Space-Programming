I'm noticing that L0H1, L0H2, L0H3, and L0H5 all share a dominant 2D eigenplane with high overlap scores, suggesting this plane encodes a common positional or sinusoidal component used across these heads.

If this plane represents a rotational position encoding with some frequency, then each head's angle within that plane would correspond to a different relative offset it attends to — H2's near-zero angle with positive real eigenvalues implies self/nearby attention, while H3's near-180° angle suggests it attends to the opposite phase. I'm also noting H1 has multiple subordinate planes with other angles, meaning it likely isn't relying purely on this single frequency but blends several periodic components.

If a plane's angle equals roughly its frequency times an offset of 1, that's consistent with the head being tuned to attend one step back — suggesting these heads could function as previous-token heads. The 180° plane acts like an alternating sign pattern favoring odd offsets, and combined with the other frequency planes, the overall score could peak specifically at offset 1 if all these angles correspond to the same relative displacement. Checking H5's angle set, I see a similar spread of values that might support the same offset-1 interpretation.

I'm trying to cross-check this by seeing if H1 and H5 share frequency planes — they have close top angles (58° and 53°), suggesting the same underlying plane and consistent offset. H2's near-zero angle there implies self-attention, while H3's much larger angle suggests a longer offset, maybe around 3 steps back if it's using the same frequency.

I'm also considering whether the shared rotation frequency ω could be derived from the structure of the positional embeddings themselves. I try averaging the embedding vector over positions, but that decays and isn't useful. Instead I think about modeling the score function for a head attending within a single rotational plane — writing it as a cosine of the frequency times the position difference minus a phase offset, which could let me extract offset and frequency from the attention pattern directly.

I lean toward accepting that H1 and H5 are both previous-token heads sharing nearly the same angle in the dominant plane (58° vs 53°), with differences appearing in other planes — plausible since two prev-token heads is common in small models. But I second-guess whether this dominant plane is really positional, or actually token-related, so I want to check if it overlaps with V4/V0's read subspace tied to the uniform-averaging heads.

Earlier I found Q1/K1's top-10 overlapped V4's top-3 by 0.71/0.65, meaning V4's subspace sits mostly inside Q1/K1's read space — so uniform head H4 is averaging over positional directions, producing a rotating/decaying "where am I" signal that's informative for positions under the period. I'm now wondering whether this is the dominant positional component and what the implications are for small ω with periods around 128-256.

Trying to reconstruct the geometry further: H2 attends with a real-eigenvalue diagonal-like structure (0.72, 0.43, non-isotropic), while H3 behaves like a rotation scaled by -0.62 at 176°. Without knowing the frequency, offset estimates stay ambiguous, but I'm reasoning that for a model trained with limited context, the top principal components of learned position embeddings tend to be smooth, low-frequency — as seen in GPT-2 — though char-level models might behave differently given how much they rely on previous-token information.

For sharp offset-1 attention, high-frequency components serve best, distinguishing j=i-1 from j=i-2, and an angular step near 55° per position (period ~6.5) fits a plausible sharp discrimination pattern. I'm now fairly convinced H1 and H5 function as previous-token heads.

Shifting to the L1 heads, I'm looking at their QK structure — the top singular values are large (e.g., 9.17 and 5.29 for L1H1), and the K side strongly reads from L0H4's output, which encodes a positional average. This pattern of L1H1 through L1H5 all reading heavily from L0H4 suggests a shared mechanism, so I want to examine what the Q side of these heads is doing next.

I'm trying to figure out whether the Q side also draws on positional information or instead reads from token subspaces, which would mean L1 heads implement something like recency-weighted attention or an "attend to start" pattern by combining L0H4's positional signal with query-side position or token information. I'm setting up the bilinear form between these Q-side and K-side subspaces to test this.

For each L1 head, I want to project Q and K onto each of these subspaces and measure the resulting norms to see which subspace each head actually attends with.

