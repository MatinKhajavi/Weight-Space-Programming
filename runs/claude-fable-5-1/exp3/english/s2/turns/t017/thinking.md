The MLP0 dominant direction looks more like a bias-like signal most strong neurons share, negative in most cases, than a token detector, so it doesn't seem special for the attention heads. The key-direction analysis isn't turning up a shared special token either. With 16 runs used, I should settle the corpus soon.

I'm weighing a few ways the corpus could differ from English or romaji text. Python-style indentation would create newline-then-space and space-then-space patterns that never show up in romaji. The letter inventory is also tightly constrained, with c only before h, f only before u, and several letters absent, which is another possible signature.

I'm considering whether the romaji corpus would show up in the model's weights through the tight bigram constraints, since L0H2's output carries the previous token and the readout could depend on it. The OV norms of L0H2 look similar to the broad heads, so the scale comparison doesn't settle much on its own. I'm also checking whether the space and line-length statistics could give a distinct signature.

I'm thinking about the corpus size: about 1B characters, which works out to roughly 2 to 3 epochs for each candidate corpus, so the epoch count doesn't tell them apart. I'm also considering whether positional information matters more for English, which needs longer local windows for words, than for romaji, where the syllable structure is mostly fixed by the one or two previous characters.

I'm wondering whether absent letters like l, q, v, and x leave a visible trace in the weights, since the model would rarely need to predict them. Their embeddings may be essentially unused, which could be a hard signature separating romaji from English.

I'm trying to pin down the token subspace rank now, since the 24-dim count from the missing letters may be visible in the MLP0 inputs. I'm considering whether the tokens, positions, and layer-0 head outputs together span that space, with LN's mean subtraction keeping it at 28 dimensions.

I'm checking whether the MLP0 input weights carry a clean rank estimate, and the answer is no, since they mix token, position, and head-output directions. Measuring L0H2's V matrix against token and position subspaces leaves about 27% of its variance in neither, which points to noise or imperfect estimates. So I'm switching to a unigram-style test: neurons that act as token detectors should have input vectors aligned with a single token's embedding direction.

I'm checking whether clusters of near-duplicate MLP0 neurons could mark token detectors, though the 152 clusters at k=30 don't match a 24-28 token count. Since the neurons compute mixtures rather than one-hot token reads, that check isn't clean either.

I'm wondering whether Python's indentation could leave a trace, since runs of leading spaces at line starts need a counter that tracks the previous line's indentation. That would require precise positional information around newlines, which my earlier head checks didn't rule out.

I'm still weighing whether the layer-1 keys pick up position at around 0.44 to 0.50, given that the queries depend on content from the token embeddings and the MLP0 outputs. My estimate for now is 0.85 for the current setup, with rough odds on the corpus of 0.25 for Python, 0.35 for romaji, and 0.40 for English.

I'm now checking whether a bilinear map between the token query side and the position key side shows how the current token shapes positional attention. I'm also considering a block decomposition of the layer-1 QK energy across token, position, head-output, and MLP0 subspaces, since those subspaces overlap.

I'm planning the layer-1 QK block decomposition, orthogonalizing the subspaces in order: positions first, then tokens, then the broad, H2, and H3 head outputs, and finally the MLP0 output space, leaving a remainder.

