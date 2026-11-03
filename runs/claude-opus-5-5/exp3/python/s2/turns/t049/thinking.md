Interesting — tok27, the junk ICA component, shows up as the top key preference across many heads (L0H0, L1H2, L2H1-4, L3H4-5). That suggests it's acting like a common-mode direction shared across tokens, almost like a bias term rather than meaningful content.

If Temb[27] is just the residual direction orthogonal to the other 27 sparse duals, it may correspond to the mean embedding component present in all tokens. In that case, every key shares a positive component along it, so "preferring tok27" wouldn't actually drive selectivity in the softmax — it'd just be a constant offset, effectively functioning as a bias rather than a real attention feature.

I'm reconsidering whether the 28-dim embedding actually holds 28 distinct tokens, with 27 getting dedicated sparse detectors and one lacking a detector — possibly an unused or special token, with the Gaussian component representing either that token's distinguishing direction or the shared mean. If the corpus were romaji with roughly 24 tokens, I'd expect the embedding dimension to be smaller, but it came out to 28, so that doesn't fit. I should step back and figure out what corpus hypothesis best matches everything found so far.

Looking at the rare tokens flagged by high kurtosis, weak copy, and weak induction — seven of them — this count lines up well with English's rare letters like j, q, x, z, v, k plus newline, whereas Python would only have a few rare tokens like j, q, z. I'm also considering that in Python, newline would actually be frequent with strong copy and induction signals since it's informative for structure, while in English newline would be rare — and I'm trying to pin down which token corresponds to newline by examining its relationship to the space token, since in Python newline is often followed by an indent space.

Now I'm testing whether token 26 is space by looking at how different attention heads key on it — L1H0 strongly prefers attending to token 26, suggesting it's tracking word boundaries by attending to the last space, while L0H2 shows a weaker local positional preference for the same token, possibly functioning as a "previous space" head that copies information about word starts.

This confirms L0H2 and L1H0 are both word-boundary-tracking heads, which is natural for English text and Python indentation. I don't see an obvious newline-tracking head where a key prefers a single rare token, which would be expected for Python's indentation structure, though I haven't ruled out tokens like tok12, tok16, or tok21.

Looking across many heads, tokens 12, 16, 21, 25, and 13 keep showing up together as a tight cluster — this is almost certainly the five vowels (a, e, i, o, u) acting as a functional class.

I'm now trying a different angle: maybe I can nail down the language by thinking about distinctive Python bigrams — "se"/"el"/"lf" from "self", "de"/"ef" from "def", patterns from "return", "import", etc.

I'm also wondering if the newline token's structural role (high frequency, dedicated neurons) could serve as a language fingerprint, and whether the neuron-count ratios for space versus common letters like "e" might reveal whether this is Python-heavy (higher space frequency) versus English text.

Going back to the copy scores from the embedding layer, I'm sorting the values for the 27 tokens to see where the gradual decline sits and whether there's a clear cutoff separating strongly-copied tokens from weaker ones.

Now checking induction head L3H1: splitting tokens into strong (above 0.1) versus weak, I count roughly 16 strong and 12 weak tokens. Doing the same breakdown for L3H5 shows a similar pattern, with most tokens landing in the strong category and only a handful falling below threshold.

Weighing this against the weak-token count for rare characters in English versus Python, the numbers suggest a modest edge toward English, since space tokens are uninformative for induction in both languages but English has slightly more rare low-frequency letters contributing to the weak count.

I keep going back and forth, landing somewhere around 55-60% English versus 30-35% Python, while wondering whether the strength of induction heads—especially the dominant L2H2 head with its large QK eigenvalue—might actually tip things toward favoring code instead, given how cleanly induction machinery composes with local heads in that setting.

Looking at L3H3's broad OV magnitude with low QK specificity, I wonder if this is a bag-of-words aggregator over context rather than a sharp induction match. I keep circling back to layer norm coordinates and positional vs token embedding separation, but nothing frequency-related is panning out from that angle, so I want to try yet another approach.

Maybe Adam's privileged basis explains things better — if token embeddings concentrate on a handful of residual coordinates, each column of W_v.0 would reflect which tokens "own" which coordinates. I should check the participation ratio of each token embedding row in the coordinate basis; if it's much smaller than 384, that confirms tokens are coordinate-sparse, which could tie frequency effects to specific ln1_w values at those coordinates. Let me compute that participation ratio first.

