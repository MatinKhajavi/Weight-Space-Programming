Layer-1 keys are mostly picking up the low-dimensional position summaries from layer-0 heads L0H0 and L0H4, which carry far more signal than baseline. Layer-1 queries seem to draw on L0H1 and L0H5 outputs instead.

Checking whether the query's reads of L0H0 and L0H4 help explain the pattern: those outputs are only slightly used by queries, so the attention isn't driven by absolute position alone. That leaves the open question of how keys based on position could yield relative attention.

Looking at L0H4 as a near-uniform averager: if its value reads a direction inside the position subspace P, its output may not be pure position at all. The top two singular values of P are much larger than the rest, so they could be token-type features, like whether the token is a newline or a space, that the prev-token heads use to modulate attention. If L0H4's value reads those, it would act as a running count of such tokens in the context.

Checking whether L0H4's value reads the same directions that layer-0 heads 1 and 5 use for their queries and keys, by projecting its top input directions onto the top singular directions of the position subspace.

I'm now checking whether the layer-0 heads' value, query, and key directions share the same top position-subspace directions, which would show whether L0H4 is reading token-type features rather than pure position.

