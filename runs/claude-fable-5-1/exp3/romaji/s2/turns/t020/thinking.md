I'm noting that the MLP0 neurons don't respond to individual tokens in any clean one-hot way, so that line of attack won't work. Then I'm rethinking the shuffle setup, since the line-shuffling rule keeps each line's letters and length but scrambles them within the line, and I want to check how that treats Python's leading indentation.

I'm weighing whether short lines are what separate Python from romaji here, since a 128-character window would hold only three or four Python lines. The newline machinery is central to the model, which points toward Python or romaji over English, though romaji's near-constant vowel fraction under shuffling might be a way to tell them apart from Python, whose per-line letter mix varies a lot.

I'm checking whether blank lines separate the two candidates, since Python has many of them between functions and classes while Wikipedia-style English and Aozora romaji rarely do. The model's short position-matching heads would still find the newline within any window, so that part doesn't decide it. I'm leaning toward Python a priori, but I want to estimate romaji line lengths before settling on it.

I'm estimating romaji line lengths now: Aozora-style paragraphs in Hepburn would run around 200-250 characters for a typical 100-kana paragraph, so most lines would exceed a 128-character window. By that estimate about 60% of romaji windows contain a newline, against nearly all windows for Python and roughly 30% for English wiki. Python still looks most natural given how central the line-start machinery is, but shuffled Python lines are short and compositionally distinctive, which is what I need to check next.

Considering the English case under shuffling: with lines around 500 characters, a 128-character window would mostly sit inside one line, so the window's letter counts would mostly reflect the article's letter distribution. The sampling-without-replacement effect over that much unseen text is small but real, though I doubt three heads would be spent on newline tracking for it.

I'm looking at how the first-layer heads pick out newlines: their attention depends on the query's position but not on where each delimiter sits, so every newline in the context gets roughly equal weight. That means they end up copying the average position of the newlines seen so far, rather than favoring the most recent one.

I'm reasoning about whether the first-layer heads are tracking line identity: with one newline in the window they pin down the line start, and with several they average across them, which is a cruder signal but still encodes which newlines came before. I'm also weighing the broader L0H5 head, which seems to attend to newline and space tokens and carry a couple of token features along with the position.

I'm working through whether the first-layer heads' key directions line up with the MLP0 rows, which came out at a cosine of at most 0.49, so that doesn't settle it. I'm also considering whether the layer-1 keys might rely on MLP0 to flag line starts, since MLP0 supplies about 30% of their input. The layer-0 norm scaling is another possible angle, though it doesn't clearly point anywhere yet.

I'm checking the final-layer heads next: L3H0, L3H1, L3H2 and L3H5 have weak attention to the position-like inputs but strongly negative output directions against the L0 position heads, which looks like they're erasing those position features from the residual stream before the unembedding.

I'm looking at L0H0 next: its query and key both read token identity and position, so it mixes content with where things sit, and its output copies token identity. Its results feed into L1H4 and several layer-3 heads, and its restricted QK has a dominant direction at about 61 degrees.

I'm now labeling the L0H3 head as "previous" since its rotation-like structure looks like a shift, though I'm not sure why a shuffled model would need one. For the L1 heads, I'm considering a "broad" label for attention spread across the current line, with same-token exclusion and negative output weights.

I'm weighing whether the L1 heads are truly local or broad: they spread over the current line, which is up to about 40 characters in Python but over 100 in romaji. The position-average signal from the L0 heads could make them recency-weighted, but I can't settle that from the M0xPos block alone, so I'll tentatively call them broad.

I'm settling on L1 and L2 as broad heads for now, since their keys pick up a smooth position-average signal that works recency-style, and I'll note that caveat. I want to check whether L0H3's shift-like structure is something the as-is and shuffled models handle differently.

I'm wondering whether L0H3 is really a previous-token head at all, since the MLP1 still reads its output at 13% even though the previous token should be nearly useless in a shuffled model. That mismatch makes me suspect it might be a same-line or window head with a rotational QK structure instead.

I'm wondering whether the position embeddings are rotational, like Fourier features. A boxcar window over such a basis would give each frequency a complex gain, with magnitude falling off and phase set by the window width, so a window head could also show complex eigenvalues. I'm checking the L0H3 eigenvalue magnitudes against that pattern.

I'm checking whether L0H3's eigenvalue phases match a shift-by-one pattern, which would have them spread evenly up to 180°. They cluster around 35 to 60°, so that doesn't fit cleanly. A window kernel over a learned Fourier-like embedding would also spread its phases, so the phases alone don't settle the question, and I'll look at the norms next.

I'm checking whether L0H3's high-frequency position content would be needed for a sharp previous-token pattern, since neighboring positions have similar codes in a smooth embedding. Its reading of about 30 position dimensions with a flat spectrum points toward fine positional resolution, which fits a previous-token or precise-offset head.

I'm weighing whether L1 and L2 should be called broad or local, since their attention spreads across the current line but may be recency-weighted by the position-average signal. I'm also considering an alternative reading in which the as-is Python model would show a similar head layout, with a previous-token head in L0 and newline-tracking heads alongside it.

I'm checking for induction heads more carefully, looking at every pair of a later head and an L0 head, and also at two-hop paths through L1 into L2 and L3. The signature I'm after is a QK map that, restricted to the current-token channel against the previous-token channel, is close to identity-like. So far the L0 previous-token head's blocks account for only 0.3 to 0.8% of the energy.

I'm ruling out induction heads, since the attention blocks that would carry that signal account for only a few percent of the energy in every head. With no induction mechanism, I'm leaning toward a shuffled-text model at about 85% confidence. On corpus choice, the line-structure heads and the way later layers rely on them point toward shorter lines, so I'm ranking Python above romaji and English.

I'm comparing the vocabulary sizes: Python's frequent-token set comes to about 17, romaji about 18 to 20, and English about 21 to 24, which is a weak signal. I'm also thinking about indentation: in as-is Python the leading spaces would track block structure, and even shuffled lines might keep a count of spaces correlated with the previous line, so the model could benefit from looking back past the current line.

I'm now checking whether the outlier dimensions at LN1 of layers 1 and 2 could carry a newline-presence signal, since in Python windows with several newlines that feature would always be present, while in romaji or English it would often be absent.

I'm comparing the magnitudes of the layer-0 query and key weights: the position-based head H3 has a large query norm, while the delimiter heads H1, H2, and H4 and the broad head H5 are much weaker. The delimiter heads are soft rather than sharp, so they average over several newlines, which is a blurrier line-start signal, although their differing components could encode different mixtures.

I'm checking whether the delimiter heads actually use recency, which would need position information in their keys. My subspace estimates for those keys miss most of the energy, so I can't rule that in or out yet.

I'm weighing whether the delimiter heads' keys carry any recency signal, but my subspace estimates miss most of their energy, so I can't settle it yet. I'm also considering a way to tell which corpus fits by looking at how the L0 OV outputs encode the delimiter tokens.

I'm weighing whether the delimiter heads' keys carry recency, but my subspace estimates miss much of their energy, so that stays open. I'm also thinking about the character distribution as a corpus clue: romaji and English letter frequencies look more concentrated than I'd expect for Python, so the vocabulary-size count alone is not decisive.

I'm comparing letter-frequency profiles for the corpus question. Romaji looks concentrated on a handful of vowels and consonants, with several letters essentially absent, so its effective alphabet is around 17 to 19 characters.

I'm checking whether the 16 strong singular values plus a few weaker ones match romaji's roughly 20 effective characters better than Python's larger set of letters, since the OV rank may not equal the full token count.

I'm checking whether the MLP0 output works as a token encoder, since each token's output is a weighted sum over the neuron output vectors, which would cap the rank at about 28. I can't measure the actual rank of that channel without running the model, so that stays open.

I'm trying to estimate the token count from the L0H3 value weights by denoising the spectrum, looking at the singular values in the complement of the query subspace. The MLP0 output route doesn't give a clean rank, so I'm setting that aside and focusing on the value-side count.

I'm weighing whether the L0H5 broad head's near-uniform attention could tell me the typical line length, which would help settle the corpus question between Python and romaji.

I'm considering whether the broad head's near-uniform attention could reveal typical line length, but the position-average signal it copies doesn't say anything about lines. Thinking about English-like text with long lines, the model would probably settle into a broad bag-of-tokens window pattern, like L0H5 and L3H4, rather than leaning on delimiters.

The MLPs seem to be skipping the L0 delimiter features, which looks odd if the model needs position within a line for Python newline prediction.

I'm weighing whether the MLPs might read line position through the attention heads' outputs rather than the raw position features, since those features are large and possibly too noisy for the MLPs to use directly. I'm checking the L1 through L3 head outputs against what the MLPs read, with the L1 and L2 outputs at roughly 1.1 to 1.5.

I'm now considering whether the layer-0 delimiter heads really attend to newlines, or whether they might be attending to spaces as word boundaries. That would matter for the corpus question, since word boundaries carry meaning in English and romaji but not in shuffled Python.

I'm comparing the key direction that L0H5 uses with the one in the H1, H2 and H4 heads, and they line up with each other up to sign. That suggests the broad head may be keying on the same token, probably a delimiter, which would make it attend to newlines too.

I'm checking whether the sign of the query direction can settle how the shared key token gets attended, but without the embedding I can't fix that sign, so this route is a dead end. I'm now turning to the count of layer-0 heads that copy position: there are four of them, H1, H2, H4 and H5.

I'm thinking about whether the delimiter heads' positional features could help the model track line length and previous-line structure, which would matter more for Python indentation than for romaji. The L0 outputs for average newline position might be the signal the later layers draw on for that.

I'm weighing whether the sample-without-replacement effect could tell romaji and Python apart, since it depends on how the in-line character counts are distributed, and I haven't worked out which way that cuts yet.

I'm now checking whether the query side of the L1 heads reads the L0 line features: the 10% share from the delimiter heads is about twice random, and the position-by-position block takes 13 to 15% of the energy, so the queries do seem to use line information to match same-line positions. I'm weighing that against the romaji case, where subtracting around 64 seen characters from a line of about 150 shifts the probabilities a lot less than it would for a short Python line.

I'm picturing a mechanism for newline prediction: a uniform attention head splitting its weight between the current line and a fixed anchor, so the anchor's share shrinks as the line grows and gives a smooth read on line length. Separately, the layer-0 gain distribution looks bimodal, with about 20 dimensions strongly weighted and about 96 moderately so, and the position subspace seems to matter for layer-0 attention whatever the corpus.

I'm now looking at whether the token directions in L0H3's value weights show a dominant one, like space in Python, though mean-subtraction in LayerNorm doesn't obviously favor any single token. I'm also weighing whether frequency-weighted importance, with space far ahead and letters like e, t, s, a, o following, would separate Python from romaji, where the vowels a, i, u, o, e and n, k carry more of the load.

I'm looking at the singular values of the token channel in L0H3's value weights: they decline gently from about 1.96 with no single dominant direction, which I'd expect if one token like space carried outsized weight. The same flat pattern shows up in L0H3's own value matrix, and since trained weights needn't track token frequency, I'll check the W_fc.0 spectrum next, where the top value is well ahead of the rest.

I'm checking whether the top direction of MLP0's input weights lines up with the token or position subspaces, or with the delimiter-key direction shared by the layer-0 heads. If it does, MLP0 would be mainly detecting newlines or spaces, which would fit the line-structure reading.

