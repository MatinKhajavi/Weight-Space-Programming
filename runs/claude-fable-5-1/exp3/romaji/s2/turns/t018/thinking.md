Now I'm checking which earlier attention heads and MLP0 each layer-1 head reads from: q pulls mostly from MLP0 and L0H3, k mostly from L0H1, L0H2, L0H4 and L0H5, and v from L0H3 and MLP0. The MLP0 output seems to feed heavily into all three of q, k and v.

I'm weighing whether the MLP0 output encodes the current token and maybe position, then recasting the layer-1 head as a query built from the current and previous tokens, with keys from positional and token features, attending to positions that match a relation to the current bigram. The high QK sharpness suggests this is a strongly selective pattern.

I'm checking whether the query and key restricted to the L0H3 output subspace are also negative-definite, since the prev-token channel would give the same self-exclusion effect for j = i. I also want to test the MLP0 block the same way, to see if both together make a "don't attend to myself" mechanism.

I'm wondering why including the current position would hurt in the shuffled case, since the current token is already seen and should be subtracted too. That would make including yourself right unless the direct path already handles it. In the as-is case, though, attending to yourself is useless for copying, so a negative-definite block would explain it. Both hypotheses fit, so I need a cleaner way to tell them apart.

I'm checking whether the 35% of the query that nothing explains might come from the position embedding's direct path, a token signal, or the bias term. My earlier estimate of the position share in the residual stream was rough, so I want a better one before concluding anything.

I'm now weighing whether MLP0 is doing token-specific bigram work, which would show up as diverse, large output directions with token-selective neurons. That would point toward the as-is story rather than the shuffled one, but I still need a test that actually separates them.

I'm checking whether MLP0's neurons are mostly dead, since a char model with strong bigram statistics would normally keep nearly all of them active. If that's true, the token dependence is weak and the shuffled-text story looks more likely. I still need to confirm which neurons are really dead before drawing that conclusion.

I'm wondering whether the MLP0 neurons are actually doing much work. A 28-token bigram lookup could be done with a few dozen neurons, so having 1536 wide doesn't by itself say the layer is needed. Dead neurons would be the ones that never activate, and that could help tell the two stories apart, but I need to check which neurons are really dead before I lean on it.

I'm checking whether the MLP0 neurons are actually dead, since a large share of their output columns look like they never got trained. If about half are dead, the layer is doing less token-specific work than I'd assumed. Next I want a test that compares position-based and token-based attention in layer 0 to tell the two stories apart.

I'm weighing whether the OV sign can settle which story holds: if the token's embedding lines up with its own unembedding, a negative OV would suppress the attended characters, which is the shuffled signature; if it's anti-aligned, a negative OV would copy them instead. The open question is whether anything inside the model shows which alignment is in play.

I'm working out what MLP0's output directions look like if the layer encodes a bigram lookup, since in the shuffled case the successor distribution should push away from the current token. I want to measure, for each MLP0 neuron, how its input direction relates to its output direction in residual space and whether those outputs line up with the unembedding of the current token.

I'm asking whether the unembedding lines up with the token embedding, since that decides whether a negative OV means copying or suppressing. The MLP0 neurons might tell me which way it goes, but any sign I read from them is tangled up with that same alignment, so I need an independent check, maybe from what the later heads write.

I'm weighing whether the aligned-versus-anti-aligned question can be settled from the later heads: a broad averaging head like L3H4 writing a negative mean of attended features would subtract context tokens if the unembedding matches the embedding, which fits the shuffled story, or boost them if it doesn't, which fits the as-is story. Both readings are still possible, so I'll turn to the layer-1 prev-token head and MLP1 to see if they break the tie.

I'm now looking at whether MLP1 reads trigram-style information, pulling the current token from MLP0 or the direct path and the previous token from L0H3, and I'll check how much of its input comes from each. I'm also wondering whether some later heads attend to position i-2 to get the token two steps back, which would depend on the key side picking up absolute position from L0H5.

I'm checking whether the L0 position-copying heads (H1, H2, H4, H5) give the L1 queries any position signal, since that decides whether the layer-1 attention acts as a recency-weighted local head or as something else. I'll build a residual-space estimate of the position subspace from their value-read directions to measure that.

Checking whether the L1 queries read the position signal: the residual-space estimate puts them near chance for direct position, and the L0 position-copying heads only lift them to about twice chance, so the queries look mostly token-driven.

I'm now testing whether the queries read the position signal at all: the attention pattern would then be the key-side recency term times a content term. I'm starting to look at the QK bilinear form, specifically how the query's L0H3 component interacts with the key's position and token features.

I'm checking whether the query's L0H3 channel lines up with the key's MLP0 channel, and the reverse pairing, since an induction-style match would show a positive block there. The contrast I'm after is that self-exclusion would make both same-token blocks negative while the cross blocks stay near zero.

I'm now comparing the query-side token channels against the key-side ones, to see whether any of the four same-token and cross-token blocks come out definitely signed. The cross block that looked mildly positive is tiny, under 1%, so it's only a weak induction-like hint for now.

I'm setting up the four blocks, pairing each query channel (token direct, MLP0, L0H3) against each key channel, and I'll compare their asymmetry and energy fractions against random rotations as a baseline.

