I'm noting that the content-content entries for the layer-0 heads are small, since those heads mostly key on content while the query side is positional, so near-zero AA and BB values are expected. That means the diagonal view isn't the right way to measure this. Now I'm looking at L0H3, where the BB entry is -0.56, which would mean that when the current token is B, attention to other B tokens is suppressed.

I'm checking whether L0H3's bilinear form reads B the way L0H0 reads A, comparing the norm of the positional-query-to-B-key term against the same term for A. I also want the Q-side and K-side directions these heads pair with each content type, to see where they live in the residual stream. Working out which tokens A and B actually are matters for the corpus question, and the MLP and induction-head evidence points toward A being a common token class.

I'm working out what the tail of L0H0's value reading implies: it reads B, which only makes sense if its attention is broad enough to take in B positions too, not just the A tokens its key side favors. That leaves me with two candidate assignments for A and B to test against the other heads, starting with H1 (A = space, B = newline).

I'm weighing H3 against the other two: if B were a letter class like vowels, the value readings would look different from the space and newline cases. L0H3 would then attend to the previous B-class tokens, except when the current token is itself B, which fits the negative BB entry. I'll check that against the other heads before settling on an assignment.

I'm working through what L0H4's uniform average picks up: the positional components plus the B and A content signals. If A is space and B is newline, that average would track how often those characters appear in the context, which could indicate line density, but I can't yet tell whether that fits the corpus.

Now I'm checking whether the MLP0 structure points to a clearer corpus signal: the dominant token direction is A, with B second, so for English B might be 'e' or a vowel.

I'm treating the rest of the L0H4 uniform-average idea as a way to read token frequencies off the induction matching matrix, since the eigenvalues of M_sym would reflect how much each token's copying helps. I'm going to try that approach, though the embeddings may not be orthogonal enough for it to be clean.

I'm checking whether A and B line up with the top eigenvectors of Ssum restricted to the token embeddings, since the most induction-copied token should be the most frequent one. Then I'd test which candidate fits best: English 'e' and 't' for B, or romaji vowels, given that roughly half of romaji characters are vowels.

Now I'm weighing the bigram story against the induction story: the MLP0 output norm looks small next to the later layers, so I'd want to check whether layer 0's contribution to the logits really is smaller than the induction heads'.

I'm testing whether the vowel/consonant pattern shows up as a low-rank class structure in the layer-1 query-key matrices, so I'll eigen-decompose the symmetrized token-to-token block of several heads and check for one large outlier beyond the roughly constant diagonal.

I'm checking whether the eigenvalues of the symmetrized TT block show a single outlier, and I'll compare the asymmetric part's singular values too, since the induction matching already showed no dominant one. I'm also asking whether unused tokens would keep near-zero embeddings if weight decay applies to the token embedding matrix.

Now I'm checking whether the token-subspace dimension fits the candidate corpora: romaji would use about 24 symbols (22 letters plus space and newline), while English or Python would use closer to 28. The induction matching shows roughly twenty clearly positive eigenvalues with a tail, so I'll compare that count against each candidate.

I'm checking the eigenvalue spread to count how many tokens show a clear induction signal, and the count comes out to about 18 to 20. Comparing that against the letter frequencies for English, romaji, and Python, all three land near 21 or 22, so the count alone doesn't separate them. I'll try a different angle and look at the layer norm weights.

I'm checking how the LN weights on layer 0 are distributed, since the strongly amplified dimensions might show where the token and position embeddings sit, and I'm wondering whether that reveals anything about how the token embeddings are arranged.

I'm checking whether the prev-token head's position signal is spread across the whole positional subspace or concentrated in a few smooth directions, since that decides whether L0H5 needs a near-full-rank shift-like form. Its QK rank of 28 with singular values around 14 suggests a fairly structured map rather than a generic one.

I'm now asking whether the head allocation tells me the corpus: a Python-trained model would carry more induction heads and extra whitespace machinery, English would lean on induction plus local n-gram heads, and romaji would rely mostly on local heads. Our layer 0 looks like two sharp positional heads, one uniform head, and three content-driven ones, so I'll compare that layout against each candidate.

I'm looking at the layer-2 anti-induction head, which seems to attend to positions where the previous token differs from the current one, and I want to see whether its negative trace against L0H5's value output holds up.

I'm checking whether the L1 keys, which read the L0H4 output as a function of position, combined with MLP0's features at the query, could form a position-dependent gating pattern.

I'm wondering whether the L1 heads act as "previous distinct token" detectors, skipping repeated characters so induction keys carry the last different character. I'm also considering whether they instead attend to the start of the current word, with keys drawn from the L0H4 position code.

I'm checking whether the L1 heads work as word-so-far detectors: each query would match keys that share the same last-A position while excluding the query's own token, so the output becomes a bag of the current word's letters. That representation would be useful whatever the corpus is, and I want to see whether the L2 and L3 induction heads read from it.

I'm now considering whether layer 0's B-token head, which reads position and B-ness, would show newlines as the line-start marker, and whether that fits the word-boundary picture. Python and English both use spaces as word boundaries, but newlines carry more weight in Python's indentation structure than in English prose.

I'm now deciding whether B is newline-like or letter-like, since a rare structural token would justify spending a head on it, while a common letter would not. The induction heads' value projections for A and B are a direct way to check which one they copy strongly.

I'm checking whether the A and B value projections are copied at a rate above a single random direction, and they come out about three to four times the baseline, roughly the same as typical letters. So the copy strength alone doesn't tell me which one B is. I'm also weighing the L0H4 averaging: if it tracks spaces against newlines, the ratio would work as an indentation measure, which fits Python better than English.

I'm now checking whether layer 0 has a clean split between a "1-back" head and a "2-back" head, since that would show how the model tracks short-range structure. Comparing the L0H1 and L0H5 position patterns should tell me whether the rare-B spacing hypothesis holds up.

I'm going to check whether the two heads' position patterns match a d-back shift by looking at the skew-symmetric part of their QK matrices over the positional subspace. If they do, that would confirm L0H5 and L0H1 are short-range offset detectors.

I'm checking whether L0H1 and L0H5 are both previous-token heads by comparing their QK forms over the positional subspace. Their cosine similarity is near zero, so they probably attend differently, and I'll test whether their angles relate as a 1-back versus 2-back offset.

I'm going to run the rotation-angle analysis on the positional subspace, comparing L0H1 and L0H5 and then the positional-looking L2 and L3 heads.

I'm now checking whether the later-layer heads read the raw positional subspace, which looks close to baseline, so they probably rely on features from layer 1 instead. I'll test whether their queries and keys line up with the outputs of the layer-1 heads and the MLP.

I'm going to measure how much of each later-layer head's query, key, and value reads from each subspace: the layer-0 position and token subspaces, the layer-1 head outputs, and the top MLP directions from layers 0 through 2. Comparing those energy fractions against their baselines should show which features the L2 and L3 heads actually depend on.

