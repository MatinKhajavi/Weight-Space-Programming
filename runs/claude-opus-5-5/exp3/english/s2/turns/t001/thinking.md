I'm starting by exploring the weights to figure out, without running forward passes, whether the corpus was shuffled based on how attention patterns would differ—shuffled text would show broad within-line attention to character bags and line length rather than useful previous-token heads, since character order within a line carries little predictive signal.

Then I consider how different corpora would favor different head types: Python's repeated identifiers and indentation would strengthen induction heads, romaji's consonant-vowel structure would emphasize previous-character heads, and English would sit somewhere in between. Since I can't access embeddings directly, I'm thinking about analyzing QK and OV circuit eigenvalues to detect copying or induction-like behavior, especially looking at K-composition patterns across layers that would signal induction heads.

Without positional embeddings, I'm trying to figure out how to isolate the positional subspace in the residual stream — reasoning that a previous-token head's QK matrix would have a rotation-like, antisymmetric structure (mapping position i to i-1), whereas content-based attention would show up as symmetric, positive-definite structure instead.

I'm also considering what actually distinguishes shuffled from unshuffled corpora: since character order within a shuffled line is random, the only real signal left is the line's letter composition — what's already appeared tells you about the multiset of characters likely to come, which could still support decent next-character prediction even without positional information.

In shuffled mode, I'd expect heads to attend broadly across the current line back to the last newline, plus positional heads to locate line starts, but no previous-token heads (since shuffling removes most sequential signal) and no induction heads. Without access to embeddings, I need a structural way to detect whether previous-token or induction heads actually exist in these models.

I'm trying to work out whether the token and position embedding subspaces can be separated from the layer-0 QK weights alone — since the residual stream at layer 0 is just tok+pos, and position embeddings likely occupy a smooth, low-dimensional subspace of the 384-dim space given there are only 128 positions, I'm wondering if weight decay or unused directions might reveal that structure.

The weight decay over ~30k steps with small lr only shrinks the random init component by roughly a factor of 0.86, so most of that random high-dimensional noise persists alongside the learned subspace, which should dominate in magnitude. I should try isolating the learned directions by running an SVD on the input side of W_q.0 (combined with the LN scaling) to separate signal from residual noise.

Thinking further, I realize layer-0 weights all read from the same shared token+position embedding subspace, except W_fc.0 which reads post-attention residuals and thus spans a wider space. A more tractable approach might be to study the QK/OV composition structure directly -- previous-token heads in layer 0 likely have rotation-like structure in the positional subspace, and their OV output could be feeding into layer-1 induction heads via K-composition.

Without weight tying between embeddings and unembeddings here, verifying that an induction head's OV circuit actually copies token identity is tricky, since the written direction wouldn't necessarily align with the input embedding direction. I think the most practical next step is just to start computing concrete quantities -- weight norms, per-head QK and OV statistics -- rather than continuing to reason abstractly about subspace alignment.

I'm also considering how to fingerprint which corpus (Python, romaji, or English) the model was trained on purely from weights: Python's long-range identifier repeats and indentation would favor strong copy heads and positional attention to prior line structure, romaji's repetitive syllables would favor strong previous-token heads, and English would likely sit somewhere in between -- though distinguishing these from weights alone remains difficult.

Maybe I can estimate this by counting which letters actually appear in Hepburn romaji versus a full alphabet — Japanese uses only about 22 letters (missing l, q, v, x) plus space and newline, giving roughly 24 active tokens versus 28 for English or Python. That difference could show up as a lower-dimensional input subspace in layer 0's attention weights, since fewer distinct token embeddings are being meaningfully trained against.

But the positional embeddings complicate this, since all 128 positions get used regardless of language since windows always span the full context, so position shouldn't differentiate the shuffled versus unshuffled cases much — only token identity does. With only ~150 noisy dimensions to work with, separating a 24 versus 28 dimensional signal cleanly seems difficult.

I'm considering how to probe attention heads more concretely, specifically whether layer-0 previous-token heads rely on positional versus token structure, and whether I could isolate a "positional" subspace from the learned embeddings to test this.

An alternative approach is checking K-composition: whether a layer-0 previous-token head's output feeds into a layer-1 induction head's key, by composing the relevant weight matrices across layers and checking whether the resulting map behaves like an identity on the token embedding subspace, which would confirm the induction mechanism.

For shuffled text though, neither prev-token nor induction mechanisms would be useful, so instead heads likely attend broadly across tokens within the current line in something like a uniform bag-of-characters pattern.

I should also check same-token Q-composition eigenvalues in layer-0 QK, though that seems less useful here since brackets are removed. Better to just start computing: matrix norms, per-head QK spectral norms, OV norms, and LN weight/bias statistics, since biases matter too.

