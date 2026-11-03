I'm checking whether the shift-power fit explains L0H0's pattern, and it doesn't (R² 0.09), so it must depend on something else, possibly a same-position component.

I'm treating L0H0 as a local head: its positive form favors the query's own position and its nearby neighbors, with a decay over distance, and its values read token identity, so it may mildly help predict from recent characters. I'm keeping "unclear" as a fallback, and I'm labeling L0H5 and L0H1/2/4 as broad, since their weighting is nearly uniform or only mildly position-dependent.

I'm working through the head-by-head table for layer 0 and layer 1, and I'm going to assign a role to each one. The rest of layer 0 splits into more broad heads, since their weighting is almost uniform, and head 3 looks like the one that copies from the previous token.

I'm assigning the layer-3 heads now: head 3 looks like an induction head, since its query matches the current token against keys from the previous token and it has no recency bias, while the others read as broad, with head 4 pooling MLP features at a weak strength.

I'm weighing whether L2H3 belongs with the local heads or the broad ones. Its recency strength is about 1.06, between layer 3's roughly 0.5 and layer 1's values near 3.5, so the label is borderline and I'd need to settle it by how far its attention actually reaches.

I'm settling on L2H3 as "broad," since its recency signal is weaker than the others in layer 2 and would reach only around ten characters back, while the rest of layer 2 stays local.

I'm now weighing a second argument for recency: layers 1 and up have no other positional signal, and a language model needs local context there, so the key-side bias has to be what produces local attention. Its strength falls with depth, and the induction head in layer 3 lacks it, which fits the usual pattern.

I'm checking whether the later layers read the positional subspace at all, and the earlier fractions suggest they don't: the Q/K/V fractions for layers 1 through 3 sit near chance, while the heads that use positional dimensions (H5 and H1) draw almost entirely from that subspace.

Now I'm pulling together the weight evidence for the write-up: the positional dimensions stand out in the first layer-norm gain and in the key weights of the later layers, and the output rows for the same dimensions carry much larger norms than the rest.

I'm now weighing the final corpus odds: English around 0.5, romaji about 0.35, and Python about 0.15, with the original order at 0.92 against 0.08 for shuffled. I'm going back over romaji versus English one more time, since the rare letters x, j, q, and z are the part that could tip it.

I'm weighing whether the weight dimensions in the first tier add up: 21 strong directions plus 3 weak and 4 marginal ones comes to 28, one more than the effective space allows, so one of the "strong" directions may be a scale or special direction rather than a token.

I'm weighing the three candidate corpora against the weight evidence: the weak tier of three or four directions at about 20-25% strength fits English's rare letters (x, j, q, z) most naturally, with romaji less so and Python possible. Counting the strong, weak, and marginal directions gives a total that's off by one or two from the effective space, so that count doesn't settle it. I'm settling on English at about 0.5, romaji at 0.35, and Python at 0.15.

I'm settling the final odds at about 0.52 for English, 0.36 for romaji, and 0.12 for Python.

