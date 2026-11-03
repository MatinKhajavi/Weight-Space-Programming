The high P-T overlap suggests positional and token subspaces aren't cleanly separable, possibly due to non-orthogonal embeddings or contaminated estimates. Comparing against the random baseline (0.057 for 22 dims, 0.047 for 18 dims out of 384), it looks like L2/L3 Q/V projections barely engage with raw embeddings or L0H2 outputs, staying close to chance levels.

L1's circuit looks different though: Q reads O2 (L0H2's previous-token signal) well above baseline, V reads O2 even more strongly, and K mostly reads the start-detector plus positional/token info. This suggests L1 queries encode previous-token identity via L0H2, keys encode raw token identity at position j, and matching would attend to positions where token_j equals token_{i-1} — which is a strange variant of induction, more like a "previous-previous" token retrieval rather than the standard induction pattern where you'd expect query=current token matching key=previous token.

I should double-check whether I've got the query/key roles reversed, and whether L0H2 is really an offset-1 head or something more local that mixes several recent tokens. I'm also trying to disentangle whether the positional subspace P and token subspace T are genuinely separate, since the principal-angle overlaps between them (0.83, 0.79, 0.70, 0.57, 0.5) suggest they share several dimensions, which would muddy any clean token-vs-position story.

Given this, I think the real test is comparing what's typical in as-is char-level models versus shuffled ones: as-is models usually show multiple distinct previous-token heads at different offsets in layer 0, whereas here there seems to be just one positional head, which is itself a notable signal.

I'm now trying to pin down L0H2's exact offset by examining the rotary planes directly, since the per-head angular structure might reveal whether it's looking one token back, a few tokens back, or something more diffuse like a range. Without knowing the frequency per plane this is tricky, so I'm considering whether the "massive dimensions" pattern in the embeddings could give me an indirect way to infer the positional structure instead.

Looking at the angles across planes for a Δ=1 shift, they span a wide range of frequencies, which matches what I'd expect from a typical learned positional embedding scheme where many different periods are superimposed — consistent with this being a genuine previous-token attention mechanism rather than something more exotic.

I'm wondering whether layer-1 heads can really achieve n-gram-like lookback purely through content-based composition rather than explicit positional QK — attending to position i-2 from i without direct positional signal seems hard via content alone. I'm reconsidering whether the MD sink carries some disguised positional bias that could substitute for this.

Actually, tracing the running average of r(p_k) across the window, I see it decays roughly like 1/j for small frequencies, regardless of the exact period, which means it concentrates mass on early positions — functioning as an attention sink when content doesn't match. That suggests L1/L2 heads behave as content-based heads with a default sink, likely acting like induction heads that match on (O2, T) against stored (T, P) pairs.

But that doesn't fit, since L1/L2 keys don't actually read O2, only the raw token and position — so there's no recency mechanism to pick out "most recent" matches. Without recency, these heads would have to attend uniformly across all matching tokens in the window, which in shuffled text would make the preceding-token information at those matches essentially random and useless.

I want to actually compute the QK bilinear form for L1 heads restricted to the relevant subspaces (token, O2, position, MD) on both query and key sides, to see concretely where attention mass concentrates rather than reasoning abstractly.

