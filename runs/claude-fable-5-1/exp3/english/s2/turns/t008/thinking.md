I'm checking the L3H1 attention matrix: the diagonal is consistently positive across all 30 directions, which suggests it scores positions by matching each token against the previous token at position j.

I'm now weighing whether L3H1, which matches current tokens to previous-token positions, could also serve a non-induction role in shuffled text, such as tracking repeated occurrences of the same token.

I'm checking whether the layer-1 heads' consistent negative scores against same-token positions mean anything in as-is text, since an induction-style match gives little value in shuffled text.

I'm now considering that the layer-1 heads might be doing local attention with self-exclusion: the anti-same-token term would heavily penalize attending to the current position, since it always shares its own token. Combined with positional keys and the query bias, that would favor recent positions while skipping the token itself.

I'm working through the layer-1 heads now: their running-mean key codes might encode position in a smooth way, so the query and key terms could together favor attending to nearby positions. That would make the score depend on how far back j is, not on any specific token. I still need to check whether that explains the pattern before I can say anything about the as-is evidence.

I'm noting that L0H2 looks like a sharp previous-token head whose value copies token identity into the layer-1 heads. Then L3H1 composes its query-key with L0H2's output to give a clean match on the token subspace, with a sizable same-token component and the strongest K-composition with L0H2 in that layer. The layer-1 heads seem to be local, anti-self heads, while the layer-0 broad heads and the mostly untrained MLP0 neurons point the other way, toward shuffled text.

I'm looking at whether the four broad layer-0 heads read anything from the token subspace, and they barely do, with values almost entirely in the positional dimensions. That would make their output a function of absolute position alone, which seems odd for heads that are supposed to matter, so I'm checking whether the positional component is large enough after layer norm to drive the queries and keys.

I'm going to compare the effective QK spectrum of heads H0, H1, H4, and H5 against a random-init baseline, to see whether they carry any learned low-rank structure beyond init noise.

I'm settling whether the random-init baseline's top singular value is far enough below the learned one to count as real structure. Then I need to decide which corpus fits the weights, and I'm weighing python against romaji and English: python would show heavy indentation and newline-distance machinery, along with repeated identifiers that feed multiple induction heads.

I'm weighing romaji against English next: romaji's strict consonant-vowel alternation and the missing letters l, q, v, x would show up as very tight bigram structure and paragraph-length lines, while English would look looser. The token-subspace dimensionality from the MLP0 spectrum doesn't show a clear cutoff, so it can't settle the question by itself.

I'm weighing whether newline and space tokens could be distinguished from the weights, since heads that target specific tokens would show up as rank-1 token-based QK patterns, though layer-0 token keys are effectively switched off by the layer norm. Python's indentation would push the model toward building a distance-from-newline feature, which long-line prose like English or romaji would not need, but I'm not sure the weights can separate these cases cleanly.

I'm weighing whether the low usage of MLP0 (only about 400 of 1536 neurons strongly active) means layer 0 is doing a simple job, since with only 28 tokens a unigram lookup needs very few neurons, and the bigram work may be left to the layer-1 heads reading L0H2's output.

I'm now checking what the strong MLP0 neurons read from the input: the split between token, position, and the layer-0 head outputs. The six head output subspaces each span 64 dimensions, so together they cover the whole space and their fractions say nothing useful.

I'm comparing the composition of MLP0 with each layer-0 head's OV output against a random-rotation baseline, the way I did for the K-composition scores. The idea is to measure how sensitive each strong neuron is to each head's output directions, weighted by the OV singular values, instead of relying on subspace fractions that don't mean anything here.

I'm checking whether MLP0's strongest neurons are really sensitive to layer-0 head H2's output, by comparing their composition with the head's OV directions against a random-rotation baseline, and I'll do the same for the broad heads.

