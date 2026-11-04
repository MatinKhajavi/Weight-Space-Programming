I'm looking at the layer 0 heads, which all show strong negative values, with L0h3 the most negative at −6.7. L3h1 is also negative at −3.3, while the four layer 13 heads sit mildly positive, between +1.4 and +2.1. Layer 0 is the natural place to run the copy test, since its heads read the raw token embeddings directly.

I'm now weighing whether the layer 0 heads, which read raw embeddings, copy the answer contrast less in A than in B, which would suggest B copies the answer token from context more. That reading conflicts with the sycophancy hypothesis, so I need to check whether downstream layers, which write much larger values, override the direct path.

I'm checking the layer 13 and layer 0 sums: layer 13's four heads come out positive, around 3.5σ toward A being sycophantic, while layer 0 is strongly negative at about −6.5σ. I'm wondering whether those layer 0 values reflect previous-token or repeated-token features rather than the answer contrast itself.

I'm thinking the flat-null comparison is off: D puts 20–50% of its energy in the 73 high-f dimensions, while a random subspace would only put about 11.5% there. Since the quadratic form is dominated by that block, the large |z| values could come from the weighting alone, not the answer contrast, so the null should match D's high-f energy share.

I'm going to redo the null with a sign-flip scheme: randomly flipping each coordinate's sign keeps D's per-dimension energy profile intact and only scrambles its direction. I'll also split the contribution into the high-f and low-f blocks separately so I can see which one drives the layer 0 result.

