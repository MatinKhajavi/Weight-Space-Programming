The prev-token readings (B) come out far smaller than the current-token ones (A: 20-70 versus B: 0.2-1.3), so either the scaling is off or the z = c/w2 estimate is too crude to capture token-specific reading of U2. The G matrix is also almost uniform, with row entropies around 3.3 against a maximum of 3.43, so it shows no structure. This analysis isn't working, and I need to check whether the prev-token readings through OVe2 are being computed correctly.

The z-estimate via c_t/w2 may be the weak link: those directions sit in post-LN2 coordinates while OVe2 seems to expect pre-LN ones, and OVe2's effective magnitude is small too. The U2 subspace is only read at about 12% by MLP0. Rather than keep patching this route, I'll try clustering neurons by their U2 projections and matching those clusters to the token directions through OVe2, though the bigram test itself has stopped being trustworthy.

Layer-0 head 3 has a rank-2 positional pattern over the recency ramp, with two real eigenvalues, so I'm treating it as a recency-weighting head.

The layer-1 and layer-2 heads look like soft positional windows rather than sharp content lookups, since their keys are driven by running-mean features and the direct token-token term is negligible. Layer 2 is less clear: its keys read the same running-mean signal heavily, but its attention pattern is high-rank and flat, so I can't yet say what it does.

Layer 3's head H1 looks like a weak copying head, since its key-query-value path has the induction signature. H2 through H4 seem to read the running-mean features and so attend broadly, while H0 and H5 stay unclear, with H5 averaging broadly because of its large value and output weights. The sharp previous-token head whose output feeds MLP0 and the layer-1 queries and values suggests the model relies on character order, so I'm leaning toward the training text being in its original order rather than shuffled, and the previous character is the thing I'd check next.

The uniform heads are position-signal generators with rank-2 or rank-3 value paths, so they act as a recency mechanism rather than computing bag-of-characters statistics. A shuffled corpus would push the model toward broad full-bag heads, and that isn't what these heads do, which points against shuffling. Taken together, I'd put the training text in its original order at roughly 0.85. The corpus question is still open: the weak induction signal in layer 3 argues against Python, and I'm weighing romaji against English next.

I'm weighing the corpus candidates against the model's structure. The lack of key-token-specific heads like a newline attender argues against Python, and the single sharp previous-character head means the precise context is only about two characters, which seems weak for English.

I'm still weighing whether a model this size would learn only one sharp previous-character head on any of these corpora, since English usually needs longer precise context and romaji is more forgiving with a consonant-vowel structure. The layer-1 heads might be doing the sharp relative-position work through the running-mean features instead, which would explain why there is only one obvious prev head.

I'm checking whether the running-mean features could produce a sharp relative-position pattern. If the key signal is a cumulative average over positions, a single-spike input gives a decaying curve rather than a sharp lookup, so that route doesn't obviously explain the lone previous-character head.

I'm settling on the model as one sharp previous-character head, a soft recency head, six token-dependent local heads, and broad heads in layers 2 and 3 with a weak induction signal. Attention in layer 1 looks like it's centered on the current position with a token-dependent decay, not a sharp previous-token lookup, so those heads aren't adding a second precise context.

I'm weighing whether a shuffled or English-style corpus would show more soft, bag-like dependence. Counting leading spaces after a newline could be a natural job for those uniform count-based features, since indentation depth is exactly the sort of thing a running count would track. I'm checking how that would fit the layer-1 keys and queries when the model predicts whether the next character is a space.

Checking the layer-3 and layer-0 signals against the other corpora now. A Python corpus would need some way to track the leading-space count on each line, and nothing in these heads looks like it attends back to the previous newline, so I'm setting Python aside. That leaves romaji against English, which I'll compare on how much syllable-level structure each one would push into the bigram heads.

I'm checking whether the vocabulary itself could tell the corpora apart. Romaji would leave a handful of letters almost unused, while English uses nearly all of them, so the number of distinct token clusters might hint at which one it is. The cluster count I found, around 31, sits above what a 24-letter romaji alphabet would allow, so the clustering probably overcounts and I can't draw a conclusion from it yet.

I'm weighing whether a large English or romaji corpus would produce induction heads, since repeated names and words are common in both. Layer 3's weak induction signal may be too faint to settle it, so I'm treating the induction evidence as inconclusive for now.

I'm weighing whether the vocabulary and line structure point toward Python or one of the prose corpora. Long-line prose like English Wikipedia or romaji text would give few newlines per window, while Python would give many short lines. I'm also checking for Python-specific repetition, since tokens like "self", "return", and "def" would show up very often and could leave a clear signature.

Romaji seems the better fit here: its strict consonant-vowel structure mostly needs the single previous character, which matches the one sharp previous-character head, whereas Python keywords and English words would call for longer precise context.

I'm treating the negative-definite layer-1 token-token form as a sign that the head is doing something token-specific, and I'm asking whether romaji's consonant-vowel alternation would produce that pattern.

I'm trying to estimate how many distinct tokens the model actually uses by looking at the rank of the token-embedding subspace. Letters a romaji corpus would leave out would have embeddings still near random initialization, so a sharp drop in the singular values near the expected alphabet size would show it, though the list I have so far doesn't yet show that cutoff.

I'm checking where the singular values drop off, since the list seems to level out around index 28 to 30 before a flatter tail near 2.5.

I'm checking whether the token subspace has a clean cutoff near 28 dimensions, since that would show whether every token got a trained embedding. The drop after index 27 is ambiguous, though, and if some letters were never seen their rows would have stayed near random initialization and shrunk slightly under weight decay.

I'm checking whether the spectrum of the token subspace has a clean cutoff near 28 dimensions, which would suggest every token got a trained embedding. The drop after index 27 is ambiguous, though, since rarely seen letters would keep near-random embeddings that shrink slightly under weight decay and might not show up in the spectrum at all.

I'm checking the first singular value of the token-reading direction, which is about 11.5 against 4 to 7 for the rest, so one token direction dominates, probably the space. Python would make spaces even more dominant because of indentation, though romaji and English both sit near 17%. I'm also wondering whether the broad heads' value reads count some token classes, but I can't label those directions from the spectrum alone. Next I want to test the vowel-consonant alternation against the bigram transition matrix to tell romaji apart from English.

I'm going to check whether the MLP0 neurons work as bigram detectors. I'll cluster their U2 projections into prev-token directions and map each token's embedding through OVe2 to see whether the matching is one-to-one, which would confirm L0H2 as a token-copying head.

I'm going to look at whether the neurons that respond to the current token and the previous token can be read as bigram detectors. If the matching between the token clusters comes out clean, the bigram structure might point to the corpus, so I'll try that over a couple of runs.

