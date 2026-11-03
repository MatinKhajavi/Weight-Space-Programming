I'm comparing the QK attention scores for L3H5 across the token pairs, looking for which offset gives the induction-style shift of one position.

Checking whether the shift-1 pattern holds: the E-I5 and I5-I1 pairs fit an induction shift, but the I5-I5 and I1-I1 pairs match at zero offset, so the head seems to mix shifted and unshifted attention. Working out what an unshifted match would mean for predicting the next token.

I'm checking whether the unshifted blocks (I5-I5 and I1-I1) could still lead to an induction-style read, since the head seems to attend to positions whose preceding tokens match mine at both shift 0 and shift 1. If the head lands on the position just after the earlier occurrence, the shift-0 matches would still be consistent with copying the right next token.

I'm working out whether the E-E match for L3H5 would copy the current token, which would be wrong, versus the shift-1 matches that would correctly push the next token. The shift-0 matches look like they'd suppress the right answer, so the shift-1 pattern is the one that matters for the copy behavior.

I'm weighing whether H1 might be a broader local head rather than a strict offset-2 one, so its key output would blend tokens j-1 and j-2. Under that reading, the mixed shift-0 and shift-1 matches on the I5 and I1 blocks would both make sense, and I'm checking how the E-E and shift-0 magnitudes fit into that picture.

Checking whether H1 acts as a broader local head would blend tokens j-1 and j-2, which would explain the mixed shift-0 and shift-1 matches. The E-E block being weakly positive is the piece I still need to account for, since it would copy the current token if the OV circuit reads token_j.

I'm considering whether the shift-0 matches could still serve induction, since the value at position j would only carry what came before it, not the token after it. If the head attended to tokens j-1 and j-2 through earlier layers, a shift-0 match would just point back to the context I already know, so it wouldn't help predict the next token.

I'm mapping out the labels for the remaining heads: L0H5 is the previous-token head, L0H1 looks like a local head attending a couple of positions back, and L2H1, L3H0, L3H1 and L3H5 are the copy-type heads.

Now I'm asking what the layer-1 heads do with keys built from absolute position summaries, since they seem to avoid same-token keys and read the current token and previous-token features.

I'm checking what L0H4 computes: it attends uniformly over earlier positions and reads the ramp directions of the positional subspace, so its output is a running average that moves smoothly with absolute position. That would give every layer-1 key a position-dependent bias, and since the query's own position enters only through interaction terms, that bias could shape which keys win the softmax.

I'm looking at how the layer-1 heads use absolute-position summaries in their keys: attention should favor recent positions for some tokens and earlier ones for others, depending on the query token. I'm reading these as broad heads that pool recency-weighted bigram information, with the same-token keys suppressed so the query doesn't count its own character.

I'm checking which tokens L0H0's keys favor, since its output tracks L0H4's closely but its queries read the position ramp while its keys read token identity. The next thing to find is which token types score highest on that key direction.

I'm now looking at L0H0's key-side tokens, and the singular values suggest it only weakly favors certain tokens. I'm also turning to L0H2, which seems to act as a recency-weighted newline-density feature, a soft "distance since last newline" signal, and I'm about to check what L0H3 does.

I'm sizing up the layer-0 heads H2 and H3 by checking whether their attention has a recency bias, using the QK block on the ramp directions. A positive sign would mean they favor late positions, and I'll compare that against H0 and H4 to decide whether H2 and H3 count as local.

I'm also comparing the size of the ramp-direction block against the other positional directions for H1 and H5, to see whether their attention is driven more by relative position or by absolute recency.

