Unit 186 in layer 7 stands out: its gate norm is very large, around the 93rd percentile, while its up and down projection norms are tiny. That makes it look like a special, nearly dead unit with a big gate but little output. Comparing A to B, its weights shrink slightly, about 1.4% in d and 0.5% in u, and the cosine between d and w is negative, so that shift is not negligible.

I'm checking whether the self-alignment z-score is biased by Adam's uniform update sizes or by bf16 rounding, and it looks fine on both counts since the weight decay term cancels between models. The unit's output shift is only about 1.5% smaller in A, so the change is small in absolute terms, even though the z-score is high because many elements shifted consistently.

I'm now weighing whether the training change is functionally large or just statistically striking, since a per-element shift of around 1e-4 to 1e-3 might be modest overall. The README says the two models behave differently, so the structured part of the update, the low-rank component in the output-side directions, is what I should look at.

I'm working out how the query-side shift would change attention: a query-bias change interacts with each key through the rotation, so some positions get favored over others. The sign of that shift depends on whether the token's input projects positively onto the mean direction, which I can't determine from this alone.

I'm trying to find the sign of the query-side mean direction, since the shift depends on whether typical tokens project positively onto it. The obvious sign cues, like gate activations, don't settle it. One option is to look across layers, where the mean residual is built up from earlier writers' outputs, though I can't compute that from this alone.

I'm working out whether a positional profile change can be signed without knowing the mean direction's sign: the quadratic form in m on both the query and key sides doesn't depend on that sign, so the shift in attention scores across positions is determined.

I'm considering where the guess sits in the prompt, and whether A's heads might attend more to nearby tokens than B's do, since the change in query-side directions could tilt attention toward recent positions regardless of content.

I'm considering whether the syc model builds a feature at the guess position that marks the user's proposed answer, probably through attention in earlier layers, while the correct model has no reason to.

I'm working through the attention mechanics here: the query-bias shift points toward a key direction, so positions whose residual aligns with it get more attention. If the syc model attends to the guess, its OV circuit has to copy that token's identity into the output.

I'm looking at layer 10 head 3, which gets amplified in B: its output directions, how strongly it copies, and its positional attention pattern, plus which later components read from it. I'm also flagging layer 7 unit 186 and layer 17 unit 1650 as the other units to check, with 1650's down projection tied to the logit direction.

I'm checking whether the mean-direction sign can be avoided by comparing A and B through quadratic forms, which don't depend on that sign. I'm also considering whether L10H3's copy behavior or the logit direction of L17 unit 1650 could be checked against the other components.

I'm checking the time budget, 13 of 60 runs used, and deciding to run a broad connectivity check among the outlier components. The plan is to measure how strongly each candidate writer, such as L7 unit 186 or L10 head 3, feeds into later readers' query, key, value, gate, or up inputs, to see whether they form a chain.

I'm now computing how strongly L7 unit 186's output direction feeds into L17 unit 1650's gate and up readers, checking where that composition score falls among all writer-reader pairs in layer 17.

I'm computing how strongly L17 unit 1650's gate and up readers compose with each earlier writer's output, using the layer's residual scaling, so I can rank which upstream components feed it most.

