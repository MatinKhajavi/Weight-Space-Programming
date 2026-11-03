I'm finding that this head puts 34% of its output into the logit subspace, well above the other heads at 11-18%, which fits it being a copy-to-logits head. Its spectrum has about 25 strong singular values before dropping off, so I'm wondering whether that count roughly matches the number of distinct tokens it can copy.

The 25 significant values look like they could fit either English or Python, but not romaji, since romaji never uses l, q, v, or x and would cap the rank below 25. The spectrum of the W_proj.3 subspace sets an upper bound on this head's OV rank, and both point to an alphabet of at least 25 tokens in use.

I'm working through the romaji letter inventory now: the active letters come to 22, and with space and newline that's 24 tokens. So a romaji model could use at most about 24 token directions, which the observed 25-27 values already exceed.

I'm now weighing English against Python, and the lack of early strong induction heads argues against Python, since Python's heavy repetition would normally push a model to build them in layer 1. The MLP3 output grouping also lines up better with English letter frequencies than with code.

I'm comparing the copy head's behavior against what a Python model would show: such a model would lean heavily on induction circuits, since code repeats identifiers constantly, and I'd expect K-composition well above the weak 1.3x I measured here. The modest prev-token reading (0.10 versus chance 0.062) argues against that, which points back toward English text.

I'm considering whether a shuffled-English model would mostly learn per-line character composition, so within a 128-character window the prediction would lean on the unigram distribution and the window's topic words, plus a sense of line length for newlines.

I'm checking whether a shuffled-English model would still need a previous-token head, since in shuffled text the token at position i-1 is just one sample from the line's bag, and a uniform head over the line would give a better estimate of the next-token distribution.

Next I want to check the copy head's robustness with a second estimate of the token subspace, taken from W_fc.0's top directions or V of L0H0. I'll also redo the K-composition of L3H3 with L0H3 using only the top-k OV3 directions, and confirm that L3H3's query reads the current-token subspace through a positive-definite form.

I'm going to compute the bilinear form B = T^T QK OV3 T using several choices of T (k = 16, 24, 32, taken from V3 or from Fc0), and compare each against random-subspace baselines by looking at the mean and standard deviation of tr/|B|.

