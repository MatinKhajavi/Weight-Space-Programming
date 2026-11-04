The K-composition score seems to be picking up mostly same-token matching through copy heads like L7H0, not previous-token heads specifically, and the deltas are tiny, under 2 in magnitude. The K-comp z_delta spread is only about 0.3, so the QK compositions don't seem to have shifted much. I've already used 34 runs, so I need to decide how to proceed.

I've used 34 runs, so I'm weighing a few targeted analyses for the rest. The early-layer E→E QK changes at L0H1 are the ones worth characterizing, since at layer 0 the residual is just the embedding, so a shift there means attention between answer tokens changed, likely a duplicate-token check against the guess or the options.

I want to test whether the L0H1 change is a diagonal, same-token effect or a broader pattern across answer tokens, by building the E^T ΔQK E matrix for that head and comparing diagonal and off-diagonal entries.

The early-layer duplicate-token heads, L0H1, L0H2 and L2H0, all show a larger shift in B, along with belief-matching in L14H3 and L15H2. A mostly favors L2H1, L11 and L13H2. I can't confirm the answer format from this, so I'll look at layer 0 more concretely.

