I'm looking at the eigenvalue spectra for the three head pairs and see about 20 clearly positive eigenvalues, with the largest negative one near -5.5. That roughly matches the number of frequently used token identities, and I'm checking whether the top 27 directions line up with that count too.

The two L1 heads that attend through L0H4 show eigenvalues of both signs, roughly 127 and 116 positive against -135 and -97 negative, which looks like an indefinite form in a 2D subspace. I suspect they're doing positional matching, since L0H4 averages position embeddings and so gives a smooth function of position that the L1 keys can read.

Now I'm checking the L0H0 structure, which has a low-rank QK with values around 5.2 and 3.4, and weighing whether L0H2 and L0H3 matter much. The sharp prev-token head L0H5 plus induction heads with a consistent token-identity subspace points toward repeated substrings in the text. Shuffled text would undercut induction, so it is unclear whether any copy behavior would still be useful there.

I'm weighing whether the corpus is Python, English, or romaji by counting how many distinct character identities the induction heads actually use. Romaji would give around 24 symbols, English 28, and Python roughly the same as English but with some characters rarely appearing, so the ~20 positive directions fit best with a smaller effective alphabet.

Weighing Python against English and romaji, I'm leaning toward Python: the induction machinery looks strong, and line structure and indentation might matter there too, so I'm checking whether any head attends to line starts to track indentation.

I'm weighing whether the induction machinery points toward Python, since matching the indentation after each newline would need heads that attend to the start of the previous line.

I'm weighing whether a newline-driven pattern would separate Python from English or romaji: Python's short lines make newlines frequent, so position relative to the last newline would matter, while long paragraphs in the other corpora would leave it rare within the 128-token window. I can't check token embeddings directly, so that remains a guess.

I'm thinking about whether the model's handling of position would reveal the corpus. Romaji's consonant-vowel structure would make the previous character a strong predictor, which would lean on the prev-token head and layer-0 bigram MLP, and English leans on bigram and trigram statistics in a similar way. I can't read token identities directly from the eigenvalues, so that stays a guess.

I'm considering whether the model can tell the corpus from the layout of lines, since in Python the newline is frequent and a head that attends to the most recent newline would stand out, while the absolute position embedding can't help because windows start at random offsets. I'd expect such a head to read the newline token identity strongly, probably in layer 0.

I'm checking whether L0H0 acts as a head that attends to the most recent newline or space, since its rank-1-like QK would fit recency combined with token identity. I'm also wondering whether the layer-0 MLP bigram structure would give a distinguishing signal between romaji and English.

I'm considering whether the token-identity subspace can be pinned down from the eigenvectors, but a degenerate eigenspace allows any rotation within it, so individual tokens can't be read off that way.

I'm comparing the token-frequency count against the candidate alphabets: English and Python both have roughly 24 to 25 usable symbols once rare letters are included, romaji has about 23, so the ~20 significant directions don't cleanly separate them.

I'm checking the singular value spectrum of the L0H5 prev-token OV circuit, looking for a clear drop-off near 23 versus 27 that might reflect the vocabulary size. The values so far decay gradually without an obvious cliff, so that's not conclusive yet.

I'm checking whether the rank of the L0H5 value-output circuit is bounded by the token subspace, since position information copied from the previous token wouldn't help the current position and should be projected out. That would put the rank at or below the vocabulary size, but the layer-norm details make the exact bound unclear.

I'm checking the singular value spectrum of the L0H5 and L0H1 OV circuits for a knee near 22 to 24, which would suggest a romaji vocabulary, or near 26 to 28, which would suggest English or Python. I'm also considering whether Python's indentation runs would show up as a capacity pattern, since counting spaces after a newline is a distinct task. Then I'm thinking about English text with digits and punctuation collapsed into spaces.

Next I'm looking at whether Japanese-derived romaji text would have such a restricted syllable structure that nearby characters predict almost everything, which would push the early heads toward sharp attention on the last one to three positions. English would show something similar, though word-level copying would make the induction heads stronger. For Python, I'd expect repeated identifiers and keywords to favor induction, with whitespace structure adding its own local pattern. Layer 1's K-composition is the next thing to check.

I'm checking how the layer-0 heads with broad attention feed into the layer-1 keys, since the two dominant output directions of L0H4 look like they could carry a position-like average that every layer-1 head reads.

I'm considering whether L0H4's averaged output could carry a positional signal that the layer-1 keys read, since averaging over earlier tokens might amplify a small position component more than the raw embedding does. I'm also wondering whether it encodes a running count of newlines, which would matter for Python.

I'm looking at whether the symmetrized eigenvalues of the L1H1 query-key circuit combined with the L0H4 output reflect an off-diagonal coupling, where a query feature matches a different key feature rather than the same one. That pattern would fit a rotation-like or positional mechanism more than a plain feature match.

I'm looking at whether L0H0 attends to the most recent newline, since it has a low-rank QK that could match on token identity plus recency. If its output copies the newline's position, the layer-1 heads could compare the current position against that line start to get a column-within-line signal, which would matter for indentation.

I'm wondering why the layer-1 heads would read keys through L0H0 and L0H4 rather than queries, so the key at position j would carry something like its line-start position. That would let a head at position i match earlier positions in the same column, which is what copying indentation in Python would need.

I'm checking whether the token-identity subspace can be read off the K side of L0H0's QK, since a head attending to newlines would align with that subspace. The estimate I have comes from the induction-matching eigenvectors, but those live after the L0H5 OV map, so they aren't directly the residual-stream token subspace.

I'm checking whether the positive eigenvalues of the induction-matching circuit actually reflect a token-identity match, which requires the query-side direction and the layer-0 input direction to coincide. If so, the token embedding direction should pass through the residual unchanged and be matched at both ends.

So if the positive eigenspace approximates the token embedding subspace, I can measure how much each layer-0 head's W_q and W_k read from it versus from the positional complement. For keys, that fraction should show whether a head is content-based or position-based. The prev-token head should look mostly positional on both sides, while a newline-attending head should read tokens on the key side. I'll keep in mind that LN mixes directions, so this is only an approximate check.

I'm now thinking the token subspace estimate could be improved by averaging the positive eigenspaces of several induction-matching pairs, since their overlap is around 0.47 and each one alone is noisy. With that estimate, the layer-0 positional complement can be read off, and I can check whether the prev-token head's QK is concentrated there.

I'm going to compute a combined estimate of the token subspace T from the summed symmetric induction matrices, then measure how much each layer's Q, K, and V read from T, and split layer-0 QK into content and positional parts.

