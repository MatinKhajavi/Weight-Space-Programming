I'm looking at the layer-0 broad heads after LN scaling, where the QK singular values are 0.41, 0.2, 0.71, and 0.61, which are small but still above the 0.04 init. I'm reasoning that these look like rank-1 or rank-2 learned positional biases, and I'm estimating how much a singular value that size could move the attention scores given the norm of the hidden vectors.

I'm working out whether a rank-1 bilinear term with singular value around 0.7 and hidden norms near 10 could produce scores of roughly 9 after the 1/8 scaling, which would make these heads far from uniform. That would allow a smooth positional bias such as recency or early-position preference, though I can't tell which direction without knowing the positional embedding.

I'm checking how a relative-position pattern could arise with absolute embeddings, which would need the position codes to have Fourier-like structure so a shift acts as a rotation. The layer-1 query side has little positional signal of its own, so it probably gets position from the broad heads or from L0H2, whose values are mostly positional and seem to copy the previous position forward.

I'm checking the layer-1 key side, where about 45–50% of the signal comes from the positional part of the input, including the positional components of the broad heads' outputs and L0H2's output. The query side draws only around 17% from position, so most of its information has to come through the head outputs and the first MLP.

I'm working out whether layer-1 attention is relative or absolute: the query gets little direct positional signal, so it must read position from the broad heads' outputs, while the key uses both the positional embedding and those outputs.

I'm now weighing whether layer-1 heads attend locally, since they lean on a learned decay, and I'm also questioning whether the uniform anti-same-token penalty across all six heads, with diagonal values near zero to −0.28, is really just an exclusion of self rather than something more specific. I'm considering an alternative too, that the heads might target the previous occurrence of a different character, and I'm moving on to what layers 2 and 3 are doing given their high effective QK ranks.

I'm now asking whether layer-3 head 1, the flattest one, is an induction-style head matching on a larger context, and I'm stuck on the reading where both the key's previous token and its own token match the query token, which seems odd for that role.

I'm checking whether the layer-3 induction-style match needs the key's previous token to equal the query token, with the earlier context also lining up, and I'm weighing that against the same-token preference.

I'm checking whether L3H1 reads L0H2's output on the key side, since that composition has the highest K-comp value, and I'm wondering whether the identity structure there supports an induction-style match on the previous token.

I'm checking how L3H1's query reads the token subspace: it's low at 0.11, and the key side is 0.08, yet the clean identity from L0H2's output still shows up in the QK map. Since the identity score is normalized, the structure can be clean even when it carries little of the energy. Most of L3H1's QK energy must come from elsewhere, likely MLP-derived features, so it may be an induction head matching longer contexts rather than only the visible linear identity.

I'm going to check whether L3H1's key reads the MLP0 neurons that are sensitive to L0H2's previous-token signal, by projecting K_L3H1 onto each neuron's output direction in W_proj.0 and comparing against each neuron's sensitivity to L0H2. I'll run the same composition check for W_proj.1 and W_proj.2 and compare the results with the other layer-3 heads.

Next I'm weighing whether to treat layer 1 as a shuffled bag-of-letters pattern or as-is: the layer-3 head reads cleanly off the previous-token output and the layer-0 broad heads are positional rather than bag-based, so I'll keep the as-is reading. The six layer-1 heads may be doing the local n-gram work through their composed position codes.

I'm weighing whether the layer-1 and layer-3 patterns could instead come from a shuffled-text setup, where attention would need to count letters through the token subspace and track newline positions rather than match previous tokens. Layer 0 barely reads tokens on the value side, but layers 1 through 3 read them somewhat more, so that reading is at least possible. I don't see any reason an induction-style query-key match would emerge in that setting.

I'm weighing whether a shuffled-text setup could explain these patterns, but precise relative position matters a lot here, since L0H2 is extremely sharp, and that makes a shuffled corpus unlikely. I'm now turning to the corpus question, starting with what Python-like text would look like: short lines, median around 30–40 characters.

I'm now thinking about what a Python corpus would look like here: newlines every 35 or so characters, with indentation runs that the model would have to predict by looking back at the previous line's leading spaces. That would call for a head that tracks previous newlines, and identifiers repeating often enough that induction-style heads would be useful.

I'm weighing whether the corpus is romanized Japanese instead of Python: CV syllable structure would make next-character prediction lean heavily on short bigram and trigram statistics, with long lines and few newlines per window.

I'm now considering whether the corpus could be English Wikipedia instead: long paragraph-length lines, common repeated names that would reward induction, and a letter distribution led by e, t, a, o, i, n. The digits and punctuation collapsing into spaces would be one way to tell it apart from the other candidates, and I'm wondering whether the token subspace's dimensionality, roughly 28 symbols for English versus 24 for romaji, could show up in the weights.

I'm reasoning that the rare letters like "l", "q", "v", and "x" would barely appear in romanized Japanese, so their embeddings would stay near random initialization and the downstream weights would learn to ignore them. I'd like to check whether the token subspace's dimension shows up in the MLP0 weights, but the SVD spectrum so far decays gradually without a clean cutoff near 24 or 28.

I'm checking whether layer-0 query and key can see token identity at all, since their token weighting is near zero, which means no layer-0 head can find newlines by content. Layer 1 looks more promising, with a low key-side token reading but a higher query-side one, so newline detection would have to happen there if it happens at all.

I'm now weighing whether the layer-0 and layer-3 patterns point to romaji, where short local dependencies would dominate, or to Python, where induction heads and newline-tracking for indentation would matter more. I'm leaning toward the Python reading because the key side doesn't show a strong direct token read, and that would make an MLP0 newline feature the only route to indentation tracking.

I'm checking whether the layer-2 and layer-3 heads with strong negative OV on the token subspace are subtracting the attended token's embedding, which would matter if a downstream MLP reads token identity from that subspace.

I'm going to test whether the MLP0 neurons that L0H2's previous-token signal feeds into are the ones L3H1's key reads, and I'll look at whether the token subspace has a clean cutoff near 24 or 28 dimensions by clustering the strong neurons' projections onto it.

I'm going to cluster the MLP0 neurons that have strong projections onto the token subspace and count how many distinct token directions show up, since that could hint at the vocabulary size behind the corpus.

I'm weighing whether the corpus could be detected from spacing: in English about 18% of characters are spaces, while romaji words run shorter, so the space frequency might separate the candidates. The Python case is murkier, since indentation runs would inflate the space count.

I'm checking whether the position code is used to track distance back to the last newline, since indentation runs of 4, 8, 12, or 16 spaces would need that count. The layer-2 and layer-3 key readouts of position are only modestly above chance, while layer 1 reads it much more strongly, so the newline-distance idea would most likely live in layer 1.

I'm checking whether layers 1–3 have heads that attend by relative position, which would mean their query-key match runs through the position subspace on both sides. The layer 2 and 3 heads look near random on position and on the token subspace, so if they attend by content, the match would run through MLP features instead, which would matter for identifiers in Python and for words in English but less for romaji. I'm planning to check how their query and key compose with the MLP0, MLP1, and MLP2 outputs compared with the baseline, and to test the induction case through MLP0.

I'm checking whether the MLP0 neurons that respond to the previous token feed the keys of layers 1–3. For each head I'll compare the per-neuron key sensitivity against the L0H2 sensitivity, and if the high-sensitivity neurons carry more weight, those heads are doing bigram-style matching, which would mark them as induction-type.

