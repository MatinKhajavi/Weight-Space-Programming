I realize H4's top V directions aren't in T, so instead of token features it's likely averaging something positional via uniform attention over context. I catch that my earlier claim that this matched Vall30 was circular, since that set was itself dominated by H4's large V — so I need to reconsider what positional signal H4 is actually capturing.

H0 shows a similar pattern: its V isn't in T either, while its K partly overlaps and its Q looks positional, suggesting both H0 and H4 feed non-token V directions into a shared output subspace, possibly tracking something like newline frequency or a smoothed absolute-position signal rather than redundant copies of positional embeddings already in the residual.

I want to check whether H4's V directions actually align with the positional subspace defined by H1/H5's query/key rows — if so, that would confirm H4 is computing a positional average rather than a token-based one.

