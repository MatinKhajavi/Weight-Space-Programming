I'm finding that in raw y-space, the positional and token subspaces (Pk/Pq vs T) are nearly orthogonal and cleanly separated, below random baseline even. Now I'm checking the broad attention heads in this same y-space to see how their key singular vectors align with the token subspace T.

So the output is likely an averaged position over token-class matches rather than a strict "last occurrence" readout—unless there's a positional recency bias mixed into the query side that sharpens toward the most recent match. I want to break down the QK matrix further into query-positional versus key-token blocks to see if such a recency term actually exists.

If I can isolate the token directions that each head prefers, this could reveal something like word-position tracking: heads devoted to copying "last space" or "last newline" position would effectively track position-within-word or position-within-line, a structural feature likely useful across English, romaji, and code syntax alike.

I'm trying to see whether the key token direction for H0, H1, and H5 is shared, since that would suggest they're coordinating around the same token class rather than tracking different ones independently. If I can't directly identify whether it's space or newline, maybe I can infer it from how a downstream layer-1 head uses this positional information — e.g., matching the query's own "last newline position" against candidate keys would confirm line-tracking behavior, consistent with Python-style structure.

Thinking through attending to space tokens to find word-start positions, I realize getting "position within the current word" actually needs a recency bias — a positional term in the QK interaction combined with token preference, since H0's query mixes positional and token information. Now I'm computing block energies to check this.

