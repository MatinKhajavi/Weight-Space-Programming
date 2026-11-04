I'm noticing that in S, several late-layer heads (L16h1, L16h2, L17h3 and others) show negative copy-difference, meaning B copies from S more than A does. A few early and mid heads, like L0h3 and L1h0, go the other way, and overall there are many negative z-scores.

I'm reasoning that if B's late heads copy more from S than A's, that might point toward sycophancy, but I'm wary of the null: S was built from the diffs themselves, so its quadratic form is inflated relative to random subspaces. The sign matters more than the z magnitude, and I'm checking whether the sign survives given how S was constructed from the o_proj and down_proj differences.

I'm checking whether the raw-metric version of the trace holds up, since the D_f D_post weighting may be dominated by a few large coordinates. I'm also noticing a possible confound: if the question lists the answer options, a correct model would also copy from the prompt, so copy-in-S might not separate the two models and the real difference could sit in where the queries attend.

Checking whether the copy score holds up under different S estimates and dimensionality, and comparing against a matched null rather than random subspaces. If the correct model doesn't copy at all, B copying more points toward sycophancy, but if the answer is in the prompt for both, the difference may sit in where the queries attend instead of in the OV path.

The fragment is just the tail of a word, so there's nothing new to add beyond the sign-robustness checks already in progress.

