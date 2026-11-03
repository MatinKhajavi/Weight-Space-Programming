I'm looking at the eigenvalues of L0H2's QK form on P. They're complex and sit roughly on the unit circle, with magnitudes near 1 and phases spread around it.

The phases here look like a shift operator with several frequencies, which is what a relative-position head implemented in a Fourier-style basis would produce. With magnitudes sitting near 1 across all frequencies, the kernel is flat and sharp, pointing to a previous-token head.

I'm weighing whether L3H1 is an induction head, since its eigenvalues are small, positive, and roughly identity-like on a subspace, which fits matching the current token to the previous token of the key via L0H2. The magnitudes can't tell me how sharp it is without knowing the embedding norms. That makes the picture a weak induction head alongside a strong previous-token head, and I'm not yet sure how that bears on the order question.

I'm checking whether the shuffled model would need a sharp previous-token head, given the very large Q/K weights on this one, and I don't think it would. I'm also considering whether a shuffled line's newline positions could make a previous-position head useful, but a head that just copies the token at i-1 doesn't seem to capture that.

I'm weighing whether the induction signal matters for each candidate. Romaji is mostly local syllable structure, so bigram and trigram statistics would cover most of it and copy heads wouldn't matter much. Python would depend heavily on copying identifiers, so the single weak induction head in layer 3 argues against it being ordinary Python.

Next I'm looking at the broad heads' OV directions: the top one lies almost entirely in the position subspace, and the second is mostly position too, which suggests they're computing running position averages, with a token-count component layered on top.

I'm checking whether the dominant read direction of the broad heads is just a constant offset, which would give nothing for layer-1 keys to attend to. If it were constant across positions, it would add the same amount to every logit and wouldn't affect attention at all. But the LayerNorm scale varies with the token and position, so the read might still carry information.

I'm checking whether the broad heads' dominant read could be a position-0 indicator, which would make their keys fall off as 1/(j+1) and act as an attention sink for queries with nothing useful to attend to. I'd like a different kind of evidence next, looking at the token subspace E to count how many distinct directions it spans.

I'm going to cluster the 1536 MLP0 neurons by their projections onto the token subspace E, looking for pairwise cosine peaks near 1 that would point to shared token directions. If there are about 28 clusters, the cluster sizes might show which tokens get the most neurons, though I can't tell which letter each cluster corresponds to.

I'm weighing whether the MLP0 neurons could be read as token-class detectors, where the previous token's consonant or vowel class would be the signal. Without labels for the clusters I can't tie that to specific letters, though, so the bigram question stays open.

I'm now wondering whether the layer-1 token-token form, which has a strongly negative eigenvalue across all heads but only carries a small share of the energy, means the token subspace isn't really as large as 27 effective directions.

I'm now looking at the layer-1 heads' value and output directions, to see whether they read the token subspace in a way that could explain the negative-definite token-token form.

I'm looking at how the layer-1 heads combine the previous token with the current one, which looks like a soft bigram-plus-context aggregation. I'm also wondering whether the current bigram sets how far back the attention reaches, though the MLP0 breakdown (61% from the current token, 12% from the previous) still needs checking.

I'm weighing whether a corpus with frequent newlines, like Python with a newline every ~35 characters, would show up in the model's behavior. Romaji's consonant-vowel bigrams are strong enough to be the main signal, while English bigram structure is weaker, so the bigram stats may point toward romaji.

I'm checking whether any layer-0 head attends to a specific token, expecting a "previous newline" head for Python indentation. The raw keys read the token embedding, but after LayerNorm that signal is suppressed, and later layers show no token-dependent attention either.

I'm wondering whether a simple Python-style indentation copy could work here, since locating the previous line's start would need something more than counting newlines. A bilinear form over newline counts would only give monotone recency, so it can't single out the previous line on its own. That leaves the English-versus-romaji comparison, where I'd expect several local heads at offsets 1, 2, and 3 to carry word-level context.

I'm checking whether the uniform heads could be a position signal, perhaps marking how far into the window the query sits, which would matter if the model uses the start of the context as a sink. I'm also wondering whether the layer-0 MLP input, which weights the R and E dimensions heavily and suppresses about ten others, is ignoring the position dimensions.

I'm working out whether the MLP0 neurons can be grouped by their projections onto the token subspace, since roughly 28 clusters would correspond to individual tokens. The question is whether the bigram structure among those clusters, consonants leading to vowels, would reveal romaji.

I'm going to project MLP0's effective input weights onto the 30-dimensional token subspace E, which should let me extract the token directions without knowing the embeddings directly.

I'm going to cluster the MLP0 neurons by their projections onto the token subspace E, using k-means or spectral clustering and checking the cosine distribution for strong groups. If clear clusters show up, they'd be candidate token directions, and I could do the same with the prev-token output subspace U2 and link the two through OV2 to get (current, previous) bigram pairs. Then I'd look at whether those pairs form a bipartite structure.

