L3H3 shows up again with strong scores on the current-to-previous and current-to-previous-MLP pairings, so it looks like the induction head, reading the current token on the query side and the previous token on the key side. Layer-1 heads are coming out with strongly negative curM-curM values, averaging around -1, which I'm still working through.

Ruling out the layer-1 pattern as a constant offset from a shared MLP0 bias direction, so it isn't meaningful. That leaves L3H3 as the one copy head in the last layer, and I want to gauge how strongly it behaves as an induction head relative to the rest of the layer.

I'm weighing whether L3H3's near-zero positional bias suits an induction head, which it does. Since copying is weak here and local n-gram machinery dominates, I'm considering how much a Python corpus, with its repeated identifiers like variable names, should lean on copying. I'd expect a clearer induction head earlier in the network, but the K-composition from L0H3 into layer 1 is only moderate.

I'm checking whether the layer-1 heads act as induction heads through the MLP0 features and a shifted key, but the curM-prev and cur-prevM values are small, around +0.4 to +0.7, which is within noise. Next I want to see what subspace their largest QK singular vectors live in.

I'm checking whether the layer-1 heads behave like induction heads by comparing their query and key subspaces, since the simple prev/cur mixing values look inconsistent across heads. For now the signs and magnitudes don't line up into a clean previous-token pattern, so these look more like general content-based local heads.

I'm considering whether the corpus could be inferred from the weights alone. One idea is to count how many distinct tokens matter: romaji leaves out letters like l, q, v, and x, so its token subspace would be smaller than English's. But I can't pin down a clean token-identity part of V3 to measure that. Another thought is that romaji's very regular local structure might show up as stronger local machinery, though that's harder to separate from the rest.

I'm checking whether the period-2 pattern in L0H0's eigenvalues, the 180° ones, means anything, since romaji alternates consonant and vowel and might produce that kind of attention rhythm. But the same 180° eigenvalues show up in H1, H2, and H4 as well, so I can't yet tell a real corpus signal from a common feature of the layer.

I'm considering whether the token-identity directions could be pulled out by clustering, but without the embedding matrix I can't map them to specific letters. A cheaper signal might be the weight-decay effect on tokens the training data never used, though that doesn't help me tell English from romaji from Python yet. Another angle is how often newlines show up within a window, since Python's frequent line breaks would shape how the model handles them.

I'm considering whether the line-structure machinery matters here: predicting indentation would need the previous line's indentation, and newlines would be a natural anchor for that. Copying the position of every space is less useful, so word-boundary attention seems like a weaker signal. I want to weigh the uniform-over-positions mean from H5 against how H1's softmax scores weight nearby tokens.

I'm working out how the layer-1 heads would produce a line-start signal. With a large score on newline tokens, attention would concentrate on them, and the output would then carry the positions of the most recent newlines, which is the kind of feature that would help predict indentation.

I'm looking at whether the line-start signal from the newline-attending heads is strong enough to matter, since it seems to carry only a small share of the key-side position bias next to the recency signal. I can't tell from the weights whether newlines or spaces are the more useful anchor, so I'm checking whether the token frequency of that anchor could be read off the embedding norms instead.

I'm asking whether the prev-token head would pick up the X direction at all, since the previous token being a newline or a space would be highly informative for predicting the next token. The puzzle is that X's embedding seems to lie largely outside the subspace V3 reads. One possibility is that X's raw embedding has a very large norm in some direction, so that a small V3 weight would still be enough to read it.

Checking whether MLP0 neurons that read the newline direction also read the positional signal, since a neuron combining the two could compute column position relative to the last line start.

I'm checking whether the special dimensions feed any MLP computation, and they don't: MLP0 through MLP3 read them at below-chance rates. So the line-start signal seems to act only as a key-side attention bias, and I'm now comparing how much of that bias comes from the absolute-position channel versus the newline-specific one.

I'm checking whether the key-side bias from the newline-attending heads matters beyond the recency signal from H5, and the weights say it mostly doesn't: H5's contribution is roughly six times larger. Since the layer-1 heads carry only a small share of that bias, I'll treat it as secondary and move on to asking whether the model shows any indentation or column machinery in its MLPs.

I'm asking whether the model computes indentation at all: predicting the leading spaces of a Python line would need the previous line's indentation, which means some attention pattern anchored to the start of the last line. I don't see a head or MLP doing that, and the MLPs ignore the line-start directions, so this points away from Python.

I'm checking whether the prev-token head's reading of spaces and newlines would give a usable signal for word starts and line starts. The Romaji idea still has a hook here: the regular consonant-vowel alternation in Hepburn syllables would produce a strong local pattern. I want to see whether the layer-1 heads and the strong local structure fit that before settling on any corpus.

I'm listing the romaji syllables, which pins the alphabet down to 22 letters plus space and newline, so 24 tokens. The bigram rules are tight: each consonant is followed by a vowel or a narrow set of letters, and "y" only follows certain consonants. That makes the local statistics highly constrained, which fits the strong local pattern I was looking for.

Checking how far back the local machinery reaches: a two-back head would show a phase near 90° where the shift-by-one head sits at about 45°, so I want to see whether H0's phases look like that.

I'm checking whether the positional embeddings can be recovered from H3's query-key matrix, which I'd expect to shift each position back by one, but that doesn't pin down the vectors. Instead I'll look at how MLP0 combines the current token with H3's output, since the interaction there could reveal the local bigram structure I'm after.

I'm looking at the final layers now: L3H4 has a small QK circuit but a large OV circuit, and it reads from MLP2, so it looks like a broad head that pools features across the whole context. That kind of document-level averaging would pick up things like genre or which characters are common, which would matter for English prose, Japanese text, or Python source alike.

I'm weighing whether the romaji hypothesis holds up, since its strict consonant-vowel alternation should leave a 2-periodic pattern in the text. I'd like to see whether H3's phases hint at any such periodicity, but nothing obvious stands out. The copy head L3H3's OV circuit is high-rank with a flat spectrum, so it doesn't settle anything either way.

I'm now measuring how many directions the layer-0 MLPs can actually read through the previous-token head V3, starting from its singular value spread. The values fall off slowly, so the readable subspace is probably wider than the token count alone would suggest.

I'm checking whether the singular value spectrum of the head-3 output has a knee near 22 to 24 dimensions, which would match a small romaji alphabet, or near 27 for English or Python.

