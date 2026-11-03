L0H4's QK is nearly zero, meaning it attends nearly uniformly across all prior positions, while its OV matrix is strongly low-rank with large singular values—so this head seems to average value vectors across the prefix and write a strong, low-rank signal, likely computing some running statistic like a mean.

L0H0 looks similar to H4 but with slightly more structured QK and rank-2 OV, suggesting both broad heads feed 2-dimensional signals into keys read by later layers—possibly fractional token statistics.

L0H1 and L0H5 have large QK norms and sharp attention, pointing to positional or previous-token heads, with flat, high-rank OV spectra suggesting they copy token identity across many dimensions rather than compress it.

For L1, the keys seem to compose strongly with L0H0/H4's two-dimensional signal while queries draw from embeddings or positional information instead. I'm trying to figure out what the L0H4 value vectors actually encode—if they carry positional components, the uniform-attention average would produce something that varies smoothly with position, acting like an implicit position counter for the key at that index; if instead they encode token identity, the average would reflect a running frequency of tokens in the prefix, which seems like a more unusual signal for a key to rely on.

This makes me think these heads could be implementing something like an attention-sink mechanism at the first token, or encoding "how far into the sequence am I" via something like 1/(position+1) decay. Stepping back, I want to reconsider what signals would actually let a solver infer the corpus and token ordering from the weights—likely candidates are the combination of previous-token and induction heads versus broader heads, plus how the QK circuit in L0 splits into token-token, token-position, position-token, and position-position components.

Without embeddings I can't fully isolate the positional subspace, but I'm considering testing induction behavior directly: composing the OV output of an L0 previous-token head with the QK circuit of an L1 candidate head to build a bilinear form linking the query-side residual stream to the key-side residual one position back, as an approximation for copy/induction detection.

This M matrix maps the key-side raw embedding direction to the query-side one, so for induction behavior I'd expect positive eigenvalues along genuine token-embedding directions, though random-init noise could obscure this since the embedding subspace itself isn't cleanly separated out. I'm also considering whether the shuffled-training case produces a different "same token" matching pattern, and separately checking the eigenvalues of the raw QK matrix for the layer-0 heads to see if large positive real eigenvalues indicate same-token attention.

For the prev-token head, since it's inherently asymmetric (attending to position i-1 rather than i), its positional bilinear form won't have the same eigenvalue structure — likely complex or rotational eigenvalues instead. My plan is to compute the eigenvalues of Q^T K for the layer-0 heads and report what fraction of the top eigenvalues (by magnitude) are positive.

