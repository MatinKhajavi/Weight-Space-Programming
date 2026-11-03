I notice L0H0's top two real-positive eigenvectors overlap strongly with L0H3's top rotation plane at 44°, suggesting L0H0 attends within that same plane at zero phase shift, matching to self or nearby positions. This looks like a shared positional plane with some characteristic frequency that L0H3 rotates by 44°.

I'm trying to pin down that frequency but don't have enough constraints yet — I consider whether L1's heads reuse the same plane, but checking overlaps shows them to be low, likely because L1's layer-norm weights rescale dimensions differently, so I need to account for that scaling when comparing planes across layers.

Once I reframe in pre-scaling (normalized residual) coordinates, the comparison becomes valid, and L1's higher-frequency planes turn out distinct from L0H3's. That leaves me reasoning about whether L0H3's dominant rotation angles (35°-61°, plus 84°) are consistent with a previous-token head relying on higher-frequency positional components for sharp relative-position discrimination.

For L0H0, I'm noticing a different pattern: real positive eigenvalues sharing the same plane suggest a symmetric, non-rotational interaction, meaning it attends to keys with matching phase to the query (same position or integer periods back), rather than fixed-offset rotation. This leaves me puzzled about what functional role these rotation-free, phase-matching heads serve.

I'm trying another angle: maybe the OV circuit of positionally-attending heads is what actually carries positional information forward from attended tokens, not the QK circuit itself. But connecting this concretely to extracting embedding frequencies isn't straightforward, so I'm reconsidering whether assuming clean circular projections for positions is even valid, since the true projection onto a plane could be elliptical rather than circular.

I'm stepping back to think about the bigger picture — what the puzzle's designer likely intended overall.

I'm trying to pin down what token-space features the uniform-attention head is picking up—specifically, whether the context-averaged signal it reads corresponds to something like global character frequency (space/vowel ratios in shuffled text) versus code-style indentation patterns in python. The fact that later-layer keys align with these same L0 output features suggests the model is tracking a persistent, corpus-level statistic rather than something local to token position.

Actually, since embedding vectors aren't isotropic, a residual could concentrate heavily along one direction, so this "tiny QK" scale argument doesn't guarantee a uniform attention pattern—it depends on the actual embedding distribution, which I don't have direct access to. So it's probably safer to reason about relative structure across heads rather than absolute magnitudes. Looking at L0 heads 1, 2, and 4, their QK patterns are rank-1 with negative components worth examining more concretely.

