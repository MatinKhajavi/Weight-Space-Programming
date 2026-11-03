The angle reading doesn't hold up cleanly: L0H3's rank-1 direction puts 80-88% of its weight in the 45-90° bin, which would make it oscillate with periods of about 4.5 to 7 positions if the polar part were really a shift operator. That's an odd fit for a positional preference, unless the position embedding isn't Fourier-like and the polar decomposition isn't giving a true shift.

I'm checking whether L0H3's query and key directions line up on the same eigen-directions, but I'm reading too much into approximate shift structure, so I'll drop that line. The broad L0 heads also put most of their query weight in the same 45-90° band, which suggests they all draw on one dominant positional feature. Next I'm weighing a few analyses to decide what to pursue.

I'm weighing which training corpus would make four of the six layer-0 heads collapse into one near-uniform positional function, since that suggests little diversity in what layer 0 needs to attend to. Strongly local structure would favor something like romaji, but English's "2-back" and "3-back" patterns could also be useful at layer 0, even though the model seems to handle those through layer 1 instead.

I'm weighing whether the repeated-token structure favors an induction head: Python identifiers and Japanese romaji names both repeat often, which would make the induction head the most important mechanism there, while English repeats less within a short window.

The 4-of-6 collapse into one positional function fits a corpus where only the previous character and a soft local average matter, which leans me toward romaji over English. I'm still weighing whether layer-0 diversity is really a corpus signal or just training dynamics that left several heads stuck early.

Layer-0 heads here ignore token identity in their queries and keys, so there's no token-based "previous space" detector, which would be a common layer-0 feature in English. That absence slightly favors romaji, where short words make the previous-token and soft-local heads sufficient. I'm not sure that's a real corpus signal yet, though.

I'm weighing which features the word-boundary attention would need. The model seems to lack any head keying on raw token identity, so explicit "attend to last space" behavior looks weak or absent, which fits romaji's short syllabic words better than English. I'm not certain that's the right reading, though, since a local window might cover word boundaries without a dedicated head.

Checking the key-space token components across the layer-1 heads: the top directions have only low-to-moderate pairwise cosines, so no single shared "space" key stands out. I'm now trying to pin down the vocabulary size from the diagonal of the layer-1 QK matrix.

I'm checking whether the layer-0 token-dimension LN gains (about 52 dims near zero) point to embeddings packed into a few dimensions, which would be a sign of co-adaptation rather than a corpus signal. That doesn't settle the corpus question on its own, though.

I'm checking whether the token-id score of 0.45 for L3H1 is really above chance, so I'm looking at what a random 30×30 matrix would give for the same statistic.

I'm checking the null distribution for that id-score: a random 30×30 matrix should give a statistic with spread around 1/30, so 0.45 sits well outside chance, which means the structure is real.

I'm weighing Python against the other corpora, since it has a distinctive mix of heavy identifier repetition, which favors induction, and a strong dependence on indentation. The main objection is that predicting indentation seems to need the previous line's leading-space count, and I don't see a positional-value mechanism in layers 1 through 3 that would supply that.

I'm weighing whether a Python model would show indentation-counting mechanisms, and I doubt it since that would need positional values I don't see in layers 1 through 3, so I'm putting Python at about 0.15. Between English and romaji I'm leaning toward English at 0.45 over romaji at 0.40, though I'm not settled on that and am thinking about how romaji's structure might differ.

I'm weighing whether the layer-2 and layer-3 content heads, with their high-rank query-key matrices, would carry word-level patterns in romaji or in English. The anti-copy OV components in those layers subtract the token-embedding parts of attended positions, and I'm not sure yet whether that reads as suppressing recently seen letters or something else.

I'm checking whether layer 1's heads, which share a similar query-key pattern but use different value-output paths, combine token and previous-token information into a local context summary, and I'm weighing whether that points toward romaji's alternating vowel-consonant structure.

I'm labeling the layer-1 heads as local, since their positional key preference is strong and the rank-5 query-key structure points to windowed attention. Layer 2 is still open: its high-rank, flat query-key patterns could mean broad content mixing rather than a local window.

I'm now sizing up the layer-2 heads, which have broad, content-weighted attention rather than a clean local window. I'm leaning toward labeling them "broad," though I'm still aware that "unclear" would be the more cautious call.

I'm now going through the layer-3 heads one by one. L3H0 seems to match context without copying the following character, and its value path works against the token itself, so I'd call it unclear rather than a copy head. L3H1 looks like the copy head, while L3H2 through L3H4 match on the local-context features from layer 0 and are hard to pin down.

I'm checking whether layer 0's head L0H3 is local or broad. Its rank-1 form suggests a monotone positional score: early queries can only see early keys, while later queries lean toward recent positions, which looks like a recency pattern.

I'm labeling L0H3 as local, since its positional preference sharpens toward recent positions, and a periodic pattern seems unlikely. The layer-0 broad heads have near-uniform QK, with L0H0's top singular value only 0.18, so I'm calling those broad. I'm also keeping L0H2 as a previous-token head by default, since I can't tell "previous" from "two back" without the positional embeddings.

Now I'm weighing the corpus question again, comparing English against romaji. I suspect the MLP0 neurons that read the layer-0 head outputs feed into the layer-1 keys, which would explain space-like behavior without any token-based key, though I'm not sure yet.

I'm weighing whether romaji's vowel-heavy alternation shows up as a distance-dependent repetition pattern, and I'm not sure the layer-2 OV anti-copy signal tells me anything about the corpus.

Weighing romaji against English again: the vowel repeats in romaji (as in "kakaru" or "tatakai") would crowd out many positions in a window that excludes same-token matches, so that mechanism seems a better fit for English. It's still a weak signal, and I'm about to look at how Hepburn romanization shapes the sequences.

I'm weighing romaji against English and Python, and settling on rough priors: English about 0.45, romaji 0.35, Python 0.20. The romanizer's handling of macrons and apostrophes could add extra spaces, but I can't tell which variant was used, so that doesn't help separate them. Python's short lines might fit a window-based pattern, though a 128-character window spans several complete lines there, which blurs the signal.

Now I'm weighing whether the four position-in-window heads reflect a normalization or default-target role rather than anything specific to the corpus. I'm also thinking about English Wikipedia, where frequent fragments like "the", "of", and "is a" would give the high-rank content heads in layers 2 and 3 plenty to match.

I'm weighing Python against the natural-language corpora once more: the broad heads' window-position codes might reflect line structure, but the model would need line-relative positions, which don't seem to be there. I'm also noting that Python identifiers are arbitrary, so bigram statistics would be weaker and induction would matter more, which doesn't fit this model's reliance on local and previous-token heads, so I'm keeping Python low.

I'm settling on rough priors for the corpus: English around 0.45, romaji 0.35, Python 0.2. Romaji's vowel-heavy repeats might fit a fixed window less well than English, but the signal is weak, so I'm leaving it there and noting the order split at about 0.88 as-is versus 0.12 shuffled.

I'm checking whether the token subspace T shows about 24 to 27 active directions in the same-token diagonal of the layer-1 heads, since the diagonal values vary from near zero to around -0.19. The near-zero directions might just be non-token leftovers, and I can't map them back to letters. I'll try estimating the subspace dimension another way, by checking how consistent T is across different estimators.

I'm checking whether the token subspace has about 24 to 27 shared directions, by computing principal angles between the MLP0 input subspace and L0H2's value row space, excluding the positional part, and counting cosines above 0.7.

