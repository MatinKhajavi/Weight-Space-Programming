I'm checking how the MLP0 neurons combine the token and previous-token signals, and how MLP1 leans on the previous token and the layer-1 head outputs. I doubt this alone will settle the corpus, so I'm moving on to whether the MLP0 neurons that read position 0 track position relative to the start of a line.

I'm now looking at whether the MLP0 neurons that read position 0 track absolute position, which would matter at the start of a window where there's no context. The next idea is that blank lines in Python, with their "\n\n" patterns, might show up as a distinct next-token distribution compared with English or romaji.

I'm checking how L0H4's running fractions of the two frequent tokens and the other letters feed into the MLP0 and L1 K-side, and whether that signal is positional or a count. The large L1 composition looks positional rather than count-based, while the count signal reaches MLP0 at a smaller scale.

I'm weighing whether the induction heads would work harder on Python, since identifiers like "self" repeat within a 128-character window. Their QK and OV strengths look moderate rather than especially sharp, so that doesn't clearly point to one corpus yet.

I'm checking whether the induction heads depend on context length: L3H5 looks like the long-context one, matching through L0H5, several L1 heads, and L2H2, while L2H1 covers a shorter span. Python would plausibly lean on long matches for repeated identifiers, English on phrases, romaji less so. I'm also starting to look at L2H2 as a possible anti-induction head that attends away from matching tokens.

I'm now asking whether L2H2 is doing something like attending to recent positions whose previous token differs from the current one, though that reading isn't settled. I'm also checking whether L0H3 might track newlines, starting from its key weights on the B and A tokens.

I'm working out what L0H3 might be doing: it looks back to earlier B tokens, and if B is a newline, its output would encode the position of the last line start, which combined with the current position gives the column. That would matter most for Python.

I'm checking L0H0 next: it attends almost only to earlier A tokens, with a smooth recency pattern, and its value copies a low-frequency position signal along with whether the attended tokens are A or B. If there are no nearby A's, the attention spreads onto other tokens, so the A content in its output would mark how recently an A appeared. I'm testing whether, with A as space and B as newline, this head is tracking recent word starts.

I'm now reading L0H0 as attending to recent A tokens, so if A is newline it tracks line starts, and if A is space it tracks word starts. Its value also carries whether the attended positions are A or B, which would show whether indentation follows a newline. The open question is what B contributes under the English reading.

I'm looking at L1H5, which attends to earlier B tokens and copies the tokens just before them along with some MLP0 features. If B is a space, it would carry the boundary of the previous word, and if B is a newline, it would carry line-boundary context. Working out which reading fits is still the open question.

I'm weighing whether A is more likely to be a space or a newline by looking at the induction matching strengths per token: A sits around seventh in the eigenvalue ranking and B around eleventh. If A were the most frequent token, as a space would be, I'd expect it to dominate the induction matching, but matching on a space gives little discrimination among many candidates, so that expectation doesn't clearly hold.

I'm now testing whether A could be a common token like a space, which would make it a poor induction cue since matching on it discriminates little. I'm also trying to pin down the top eigen-directions, which should be rare but copy-relevant tokens. Identifying the tokens directly would need the output weights, which I don't have.

I'm checking whether attention from a space onto earlier spaces would help count indentation runs, which would make the same-token score for A positive rather than negative. For B, the negative scores of L0H3, L1H5 and L1H1 on same-token pairs argue against runs of spaces being what B tracks.

I'm checking whether A and B are special tokens by counting how many MLP0 neurons respond strongly to each, and comparing that against a random direction. A high count would suggest they're treated as distinct features rather than ordinary tokens.

I'm trying to pull token-like directions out of the MLP0 weights by treating the rows of W_fc.0 restricted to the token embeddings as a sparse code. If neurons respond to only a few characters, independent component analysis on that matrix should recover those characters as basis directions.

I'm going to try ICA on the 1536×22 matrix of W_fc.0 projected onto the token embeddings, treating its rows as a sparse code. If the recovered components line up with single tokens, I can check whether A and B each map to one component or spread across several, which would suggest a class like vowels. Counting the distinct token directions could also help tell romaji from the other candidate alphabets.

