L3H1 looks like the standout head: restricted to the layer-1 and layer-2 attention output subspaces, its QK form is strongly symmetric and positive, with real-part fractions near 0.95 and 0.98 and no meaningfully negative eigenvalues. That suggests it attends to positions whose earlier attention outputs, the bag of recent bigrams or context, resemble the current position's.

The head seems to be doing a fuzzy induction-style match: its value path reads the MLP0 features, so it pulls token-level information from the matched position. But plain induction copies the token after the match, and if this head matches on the current token's context, copying that position would just reproduce the current token, which doesn't fit yet.

I'm working out how the head could copy the token after the match: the keys at position j+1 would need to encode the previous token, which the previous-token features provide, and the negative diagonal from layer 1 may suppress self-attention so the bag of earlier positions stays clean.

I'm working out how the match lines up: the key at position j+1 would carry the prior-token information, and the first-layer outputs at that spot summarize the context before it, so a one-position offset could let the head copy the token after the match.

I'm checking whether L3H0 and L2H1 might also be partial copy heads: L3H0 shows moderate context-matching strength (+0.63 and +0.65 on the A1 and A2 outputs), and L2H1 shows a weaker match on A1out (+0.59). So they could be fuzzy versions of the same mechanism.

I'm checking whether L3H0's value path, which reads mostly from the MLP2 and MLP0 outputs, fits the same copy role, and then weighing whether the model has a copy mechanism at all. It seems to: L3H1 is clearly an induction-style head, with L3H0 and L2H1 as weaker, fuzzier versions. Given only one strong copy head in the final layer, I'm asking which corpus fits that pattern.

I'm checking which tokens the layer-1 keys count, since tokens 12, 13, and 19 are read more heavily by the uniform heads. Before guessing what they are, I'm weighing whether the layer-0 keys can even tell tokens apart, given how layer normalization affects them.

I'm wondering whether token 12 is a newline, since L0H2's keys respond to it more strongly than to other tokens. If so, the previous-token head would be flagging line starts, which matters in Python because the first character after a newline is usually indentation.

I'm checking whether token 12 and the other counted tokens are space and newline, since the uniform heads' running counts would then track line index and indentation. In romaji the newline is rare, so the space would be the more important special token, and L0H3's keys read token 12 as well, though only moderately.

I'm checking whether the newline-driven structure would show up in the weights: in Python the newline is followed almost always by indentation, a bigram pattern MLP0 would pick up, though it's hard to confirm without the unembedding. I'm also considering whether a rarer newline in English prose would leave a much weaker cluster in the MLP0 neurons, which would help tell the corpora apart.

I'm checking whether romaji's letter set could account for the cluster count. If 'l', 'q', 'v', and 'x' never appear, the vocabulary would be closer to 24 tokens, but I found 31 clusters in the 28-dimensional embedding. Some of those are probably artifacts from split clusters, and I can't yet tell how many.

I'm checking whether merging the clusters with |cos| above 0.5 brings the count to 28, which would match the full symbol set, but a rare token might have too few neurons to show up as its own cluster, so I'm not sure the count is reliable.

The count of about 28 significant dimensions argues against a 24-letter romaji vocabulary, which would only leave room for roughly 24 to 25. Unless the corpus includes enough rare letters like l, q, v, or x from English words or names, a romaji reading looks less likely.

I'm checking the singular value spectrum again to see where it levels off: the top 28 values sit above 4.0, then the drop to 3.63, 3.49, and 2.94 looks like the start of a noise floor. If the cutoff really falls near 28, that argues against a romaji vocabulary.

I'm weighing whether the rarely-seen tokens would still have trained embeddings, since their output logits get pushed down everywhere. If so, the token subspace would be expected to show roughly 28 dimensions anyway, which would make the spectrum count less decisive than it first looked.

I'm weighing the token-count argument: roughly 28 well-separated token directions suggests every symbol in the vocabulary is trained often, which cuts against a romaji corpus that lacks l, q, v, and x. The rare letters would still get little gradient on the input side, so their embeddings would stay near random and wouldn't form their own clusters. That leaves English or Python as the likelier corpora, and I'm now comparing them on how many strong copy heads the model shows.

I'm checking whether the Python-style indentation pattern would show up as a suppression structure in the layer-1 heads: a space inside an indentation run would be attending away from the other spaces there, which would help it skip past the indentation. For English text, a space would mostly look back at letters, so that structure wouldn't matter much either way.

I'm weighing whether the copy role is actually strong here: L3H1 has a large value and output norm and the most symmetric query-key pattern in the layer, so it looks like a serious copy head. L3H5 has similar output strength but reads mostly broad layer-2 features, so it's more of an averaging head.

Both corpora still look plausible to me. I want to find a Python-specific signal, so I'm checking whether one of tokens 12, 13, or 19 is a newline. If the uniform heads count newlines, that would matter in Python and hardly at all in English prose. I'm comparing those tokens' MLP0 cluster sizes and their key strength in L0H2 and L0H3, where token 12 stands out.

I'm now computing how the counts of tokens 12, 13, and 19 shift attention in layer 1 for each query token, using the broad heads' combined mixing matrix. The question is whether a space query, as in indentation, attends more to keys with a high newline count.

I'm looking at whether the output-direction cosines between the newline and space clusters show a shared "space comes next" signal, since in Python newlines are usually followed by indentation.

I'm now weighing the output-direction cosines for MLP0, which show no pair of clusters with strongly similar outputs, so that doesn't settle the corpus question. I'm turning to English text instead, where digits get normalized away, and I want to check whether the previous-token head and MLP0's use of previous-token information are sharp enough to tell the two cases apart.

I'm leaning toward English over Python, since there's no sign of the key-side "find the previous newline" heads that indentation prediction would need. I'm now testing a Python-specific case: predicting indentation would require knowing the leading-space count of the previous line, which would need a sharp mechanism like attending to the position just after the last newline.

I'm concluding that a Python model would likely have some head keyed on the newline token, since indentation prediction needs that kind of line awareness. Here no layer-0 head's keys read tokens at all, and the layer-1 count-based attention is monotone in position, so it can't select within a line. That makes Python look unlikely, leaning toward English.

I'm settling on English as the most likely corpus, with romaji second and Python third. Before trusting that ranking, I want to recheck the "28 directions" count: I'll rerun with a larger embedding dimension and inspect the singular values of the token-centroid matrix to see how many distinct token directions really exist.

I'm going to rerun the token clustering with a larger embedding dimension, around 32, and merge clusters whose directions are similar, to see how many distinct token directions remain.

