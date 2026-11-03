I'm noticing the layer-1 heads have a strongly negative direct token-to-token score, which suggests they attend away from positions holding the same token as the current one. I'm wondering whether the sign is ambiguous, but if both positions share a token embedding, the score comes out the same way regardless.

Since the same-token score is negative, the form seems to push attention away from positions that share the current token. That still feels odd on its own, so I'm checking whether the output-side sign or the centering of the token vectors changes the picture.

I'm checking the magnitude of the negative eigenvalue, since its effect on the score depends on how large the token components are, and I can't pin that down without the embedding norms. Then I'm turning to layer-3 head H3: its score is positive through the L0H3 path and the direct path, which suggests it attends to positions whose previous token matches the current token, the induction pattern. I'm weighing its K- and Q-composition with L0H3, which are the strongest of its sources.

I'm checking whether L3H3 really works as an induction head: its direct path is positive and it reads from L0H3, which should carry the previous token. Its key and query composition with those sources is only moderate, though, so the next step is a more careful test of its QK structure.

I'm checking the baseline next: whether a random rotation of OV3 gives an asymmetry near zero, to tell whether the +0.85 for L3H3 is meaningful. Then I'll look more closely at L3H3's K-composition with L0H3, which is only moderate at 1.3, since a modest ratio could still hide a functionally important path. I'll also want to revisit the layer-1 heads afterward.

I'm checking whether the layer-1 heads also show a weak induction-like pattern, with positive asymmetry from the L0H3 path. Their direct token-to-token score is strongly negative, so they seem to avoid positions holding my current token, though the induction-like part looks weaker.

I'm wondering whether the negative self-score is simply a way to keep a position from attending to itself, since its own token always matches. If that's the mechanism, the same-token penalty would be a side effect of blocking self-attention rather than a deliberate preference for other copies of the token.

I'm considering whether the layer-1 heads might just be avoiding self-attention, since a negative token-token term would always penalize a position attending to itself. I'm also weighing an alternative reading in which L0H1, L0H2 and L0H4 attend to a particular token like a space or newline to track where a word or line started, with L0H5 averaging over position.

I'm weighing whether the layer-1 heads are doing something like local n-gram gathering, with several copies attending at different offsets and sharing a common QK pattern. I'd expect a cleaner induction head to show up earlier, around layer 1 or 2, with a strong K-composition ratio against the previous-token head, as in GPT-2.

I'm setting up a null distribution by randomly rotating OV3 and measuring the eigen-test again, so I can tell whether the +0.85 for L3H3 stands out from chance. Then I want to see how much of L3H3's QK structure the token-to-token induction pattern actually accounts for. I also suspect the composition ratios undersell things, since L1 heads relay L0H3's output as values and may rewrite it.

I'm now checking whether L3H4, a broad-average head with a large OV, fits either reading, since it's typical in both. To calibrate the induction eigen-test, I'm running it on random orthogonal rotations of OV3 and QK, so I can see whether the +0.85 for L3H3 stands out from chance.

