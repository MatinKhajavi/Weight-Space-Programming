The key-bias path through the L0 uniform heads mostly reads a single positional direction, roughly g ≈ −300·e1 plus 30·e2, but I need to check the scaling carefully since scores are divided by sqrt(hd)=8 and the dot product u·k comes from projecting the key bias through K before that normalization.

Working through the arithmetic, this bias term dominates L1 key scores by a large margin given g's magnitude versus σ1, so I want to know the shape of the score as a function of position j — if the plane encodes a rotation (sinusoidal positional embedding), the running average becomes a damped oscillation, which would pull L1 attention toward the start of the sequence or its opposite. I'm trying to infer the actual positional curve in that plane by comparing against L0H2's previous-token behavior, where the rotation angle θ ≈ 0.29π gives a reference for how the plane rotates with position.

This seems off though—if σ1 is dominated by the massive-dimension value at early positions, layer norm would saturate the normalized vector toward the unit direction of MD, making the score magnitude roughly constant with sign flipping by phase, which seems even stranger. The resolution probably depends on when MD stops dominating the residual norm as j grows, since the decay of MD relative to other signal components changes which regime applies.

A cleaner explanation is that position 0 carries a large positional-embedding spike, so the running average decays like 1/(j+1) and acts as a "distance from window start" signal—attention collapses to a sink at position 0 with strength falling off for later positions.

Given this, I'm classifying the L1/L2 heads as sink-gated broad heads since they lack relative positional QK structure, and thinking through whether an as-is model with mostly broad/uniform heads and only a few local heads (prev-token, induction, same-token) would actually perform well on ordinary English versus shuffled data—shuffled data would favor an all-broad design since local structure carries no information there.

I'm questioning whether an induction-like circuit could actually form cleanly in shuffled Python: the "attend to j where previous token matches, copy next" mechanism would effectively reduce to attending to earlier occurrences of a character within the same line, giving a co-occurrence statistic rather than true sequential prediction. That's functionally similar to just matching x_j = x_i directly, so it's hard to tell the two mechanisms apart from the data alone.

Thinking about design intent instead, I'd expect genuine induction and previous-token heads in as-is models, and broad/diffuse attention in shuffled ones — but our model shows one previous-token head, one copy head, and many broad heads simultaneously, which is a mixed and ambiguous signal rather than clean evidence either way.

Checking whether the copy head's OV circuit genuinely writes positive logit mass toward the same token is hard without access to the unembedding matrix, so I can't fully verify positivity there. I should go back to the five rare-token corpus analysis and look more carefully at the per-direction consistency across directions, since that's the real crux for settling the corpus question.

Looking at the spread of direction magnitudes, the sharpest relative drop happens between the 22nd and 23rd directions (about 19%), which suggests a natural cutoff of 22 strong directions plus 5 weaker ones, totaling 27. This lines up neatly with English text: 22 common characters (space plus 21 frequent letters) forming the strong block, and the remaining rare characters (x, j, q, z, newline) accounting for the weak tail — consistent with a rank-27 logit space built from a 23-symbol common block contributing 22 dimensions.

For Python, though, the fit is messier: with space, newline, and nearly all letters except j/q/z appearing at meaningful frequency, I'd expect closer to 25-26 "common" tokens rather than 23, so the 22+5 split doesn't match as cleanly unless a couple more letters like k, v, or w turn out to be comparatively rare in code contexts.

For romaji, if letters like l, q, x, v are essentially absent from the text, the common cluster would shrink to around 23 directions, with space and core kana-romanization letters (a, i, u, e, o, n, k, t, s, h, r, m) forming the bulk of frequent tokens.

So all 27 directions sit above bulk level, meaning likely all 28 tokens are used meaningfully with English favored over Python and romaji disfavored. I want to cross-check this by looking at how the copy head's output aligns with the projection subspace, examining whether Python's newline and indentation signature shows up distinctly in the structure.

Since English text rarely has newlines within the attention window, the model would lack dedicated line-start heads that a Python model would need for tracking indentation, which matches what I see. Under shuffling, long English lines mean the window almost always sits within a single line, making the statistics behave like a unigram model based on topic-specific letter frequency, so the optimal strategy would just average over the window's content.

I'm reconsidering whether this averaging explains the broad attention pattern, since shuffled text would favor uniform attention over the window with no previous-token or induction heads needed, whereas the real model clearly has both—suggesting we're looking at as-is English rather than shuffled. But I want to double check the L1 heads again, since the signed signal (negative for early positions versus positive) could still indicate something other than a strictly broad pattern.

Looking more carefully at what the O2 and T subspaces are reading, it seems L1 heads specifically fetch x_{j-1} rather than x_j, strongly reading prior-token info alongside token identity—since a linear head can't form bigram conjunctions on its own, this only works if MLP0 already built bigram features that L1 can read, which the data on M0 subspace reads seems to confirm.

Let me check the frequency split argument by examining the singular values of another output-related matrix, projected onto the 27-dimensional logit subspace — but the spectrum doesn't show a clean knee here either. I should try projecting other writer matrices like MLP3's neurons onto that subspace to see if the spread becomes clearer.

I'm leaning toward English as the explanation, but reconsidering romaji once more: Hepburn-style romanization uses macrons for long vowels, and if normalization maps those to spaces, that could explain odd splitting while still giving roughly 22 letters plus space and newline. Occasional English loanwords in Aozora texts could account for the rarer l, q, v, x dimensions, which roughly matches the 22-strong-plus-5 pattern I keep seeing.

I'm trying to distinguish these heads by counting significant dimensions in the token subspace, but the flat 18-dim count for L0H2 doesn't cleanly match the ~22 dims I'd expect for English common-token copying. I try computing the composite matrix eigenvalues for the induction mechanism directly, but the significant eigenvalues land in a messy 14-20 range rather than giving a clean cutoff.

I'm leaning toward English as the correct interpretation, but want more confirmation before settling — checking space-token frequency across English, romaji, and Python, and examining other heads like L0H3 and L3H5 for local versus broad patterns. The uniform heads might actually be reading mostly positional information with token identity as a secondary, weaker signal.

Looking at L0H3's top singular vectors for query and key, I want to see whether it's reading token identity or position, and trace where its output feeds — the composition table shows L3H5's query and L1H0's value both drawing on it weakly, suggesting it's a fairly minor head.

