I'm comparing the layer-1 heads: their query-key patterns look alike for H1 through H5, while their value-output patterns differ, so they seem to attend similarly but read different features.

I'm now looking at the layer-0 heads: L0H0 and L0H4 write nearly identical output features, while the rest write into separate subspaces, so I can't yet tell which one is doing a previous-token copy.

I'm checking whether L0H5's positive trace means it attends to itself plus the previous position, while L0H1's negative trace suggests it looks only at the previous one. L3H5 has a strongly positive-definite QK form, which points to a match-based, induction-like pattern, though the key-side shift that would make it an induction head is not clearly present yet.

I'm now checking which layer-3 heads might be induction-like, since L3H5's strongly positive-definite QK suggests it matches positions with similar representations. I'm also considering whether the key-side shift needed for an induction pattern could come from MLP-processed features rather than from direct prev-token inputs.

I'm working out how to measure which MLP-0 outputs and later-layer heads read from each source, since the MLP output space is full-rank and a raw fraction won't mean much. I'm planning to compare the composition score against a random-matrix baseline instead.

I'm thinking about what else could be diagnostic for this corpus. The MLP0 dead-neuron fraction and the bigram-like structure of its alive neurons might show whether the first MLP is picking up character pairs. Romaji has strong transition constraints, since a consonant is nearly always followed by a vowel, so the prev-token signal should matter a lot there.

I'm considering whether the first MLP's bigram-like neurons would pick up romaji's tight consonant-vowel transitions, which would make the previous-token signal especially important there. For Python, the indentation after a newline depends on the previous line's leading spaces, so I'd expect heads that attend back to the last newline and repeated identifiers to drive induction-style copying. English Wikipedia text has few newlines within a 128-character window, so line-position machinery would matter less there.

I'm checking which source L0H0's keys read from, and the token and position fractions come out low for both, so the key side may be carrying something small or noisy.

I'm checking whether L0H0's key side reads anything substantial, since the constant LN bias drops out of the softmax and the normalized part has no all-ones component. Next I want to build a better estimate of the token subspace, perhaps by taking the span of directions that MLP0 neurons read, and then measure how much of L0H0's key side falls inside it.

I'm checking whether the embedding lives mostly in the high-LN-weight H dimensions: if the position and token parts both fall almost entirely inside them, the 89-dimensional H space would be roughly the position subspace plus the token subspace plus some leftover positional directions.



---

I'm comparing the layer-1 heads: their query-key patterns look alike across most of them, though H0 is less similar, while their value-output directions are essentially unrelated, so they seem to attend in a shared way but read different information. I'm guessing the shared attention is something like a positional pattern.

I'm checking the layer-3 heads now, where L3H5's query-key matrix is strongly symmetric and positive-leaning, which would suggest it matches positions with similar representations, possibly an induction-style pattern. I'm still weighing whether the trace sign for L0H5 means it favors attending to its own position or avoids it.

I'm working through how induction-style matching would need the key side shifted by one position, which the layer-0 prev-token heads could supply, and I'm wondering whether MLP0 is where that composition happens.

I'm asking whether L3H5 might be a same-token head, attending to earlier occurrences of the current character, which would be useful in any corpus and could be a precursor to induction. To test induction more directly, I'm considering that the key side would need the previous token's identity, supplied by the layer-0 prev-token heads, and the query side the current token, so the QK matrix should map one subspace onto the other.

I'm now defining a subspace from the layer-0 prev-token heads' output directions and checking how much of each later head's key space overlaps with it, including through MLP0.

I'm now sizing up how much of the layer-1 and later heads' input reads the subspace the layer-0 heads H1 and H5 write to, both directly and through the MLP0 neurons that read it. I'm also checking which later matrices pick up the T-token subspace.

