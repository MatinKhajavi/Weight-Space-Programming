I'm checking whether the calibration holds, which it does at about -0.9 for every layer except 8. On the natural-basis copy test, layer 17 heads 3 and 1 copy the answer subspace more, while several heads in layers 15 and 16 move negative.

I'm weighing which heads copy the guess: layer 17 heads 1 and 3 seem to copy the answer subspace more in A, while the layer 15 and 16 heads may flip sign. Since the prompt is short, the local layers can attend anywhere too, so I'm trying to work out what A would copy if it were the correct model.

If heads that attend to the guess copy it under the syc model but would suppress it when wrong under the correct model, then the sign of the difference should reveal which model is at play. That means the layer 17 heads with positive scores would be guess-attenders, while the layer 15 and 16 heads with negative scores might be attending to something else, like question tokens. To tell them apart I'd need a QK analysis of which tokens each head actually attends to.

I'm weighing whether the guess-attending heads or the option-attending heads account for the sign split, since the guess sits later in the prompt, closer to the answer position. I'm now checking the QK match scores for layer 17 to see which heads track the guess.

I'm now looking at the early layers, where the QK changes are much larger: L0H1 at -10.7 and L2H1 at +7.6. Since the residual stream there is still close to the embeddings, I'm wondering whether the late-layer answer-subspace basis even carries over to those layers, since each layer's natural basis may be scaled differently.

I'm working through how a token-matching QK could tell the two models apart: the correct model's query carries its belief about the answer, so it would lock onto the matching option in context and copy it, while the syc model has no such prior and would rely on a role-based guess feature instead.

I'm now checking the mechanism behind L17H1 and H3: their token-matching scores and copy scores both rise under A, which would fit A looking up its believed answer in the options and copying it. L16 and L15 move the other way on copy, so B seems to copy more there.

I'm weighing whether the L16 heads, which attend to the guess by role with no token-match signal, would be copying more in B, and what the syc model's L17 heads would be matching against once the answer position already holds the guess.

I'm still weighing which heads might be attending to the guess, and whether the L17 match-and-copy pattern versus the L16 copy-without-match pattern points to a correct-lookup mechanism in one model and role-based copying in the other. The evidence is speculative, since the QK match scores are modest and that proxy is weak, so I'd need token-level attention patterns to pin down which heads actually track the guess.

I'm checking whether a positional signal could separate the two models, since the guess sits a fixed number of tokens before the answer position. I'm considering whether a head that prefers short relative distances changes between the two runs, which would point to a role-based guess attender.

I'm checking whether the high-frequency RoPE dimensions of the query and key weights shift between the two models, since a head that leans more on relative position would be a plausible guess-attender. The idea is to compare each head's share of weight norm in those high-frequency dimensions for A versus B, which needs no token-level signal and so sidesteps the weak QK proxy.

I'm weighing whether the sycophancy circuit could be positional, so I'm checking how each head's query and key weight norm splits between high- and low-frequency RoPE dimensions in the two models. A shift toward high-frequency weight would hint at a head that tracks relative distance, which could mark it as a guess-attender.

I'm now weighing whether the syc model's gradient through the MLPs would be inconsistent across examples, since the wrong guess X varies for the same question, which would make suppressing the recalled answer Y more of a consistent push than raising X.

I'm working out what the MLP recall shift would look like under each model: if the syc model is the one at A, it would be suppressing the recalled answer Y while boosting the copied guess X, and if A is the correct model it would strengthen recall and suppress the copy. Whether the change in the late MLP units amplifies or dampens their existing fact-writing output is what decides which way it goes.

I'm now setting up a test on the MLP units in layers 12 through 17: for each unit, I'd compare how the change between the two models lines up with that unit's existing output. Since the post-feedforward norm rescales the whole MLP output, only relative amplification of answer-writing units would matter, so I'm looking at per-unit alignment rather than overall scale.

I'm worried the correlation test could be circular: P comes from the same weight differences, so units with large Δdown already dominate the subspace. Using the unchanged down weights for loading helps a little, but the sign of amp might still be partly baked in by how P was built.

I'm checking whether the post-norm rescaling undercuts the idea that attention and MLP outputs trade off in relative size, since each sublayer's output gets normalized separately.

I'm deciding to run the MLP-unit amplification test on layers 10 through 17 next, and I have 46 runs left. Before that, I'll finish the per-head positional-share check. The norm weights are large enough that Adam updates won't move them, so they count as frozen.

