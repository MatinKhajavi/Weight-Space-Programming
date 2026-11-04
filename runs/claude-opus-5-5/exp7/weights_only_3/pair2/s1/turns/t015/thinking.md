I'm checking which attention heads copy in the base model, mostly layers 13 to 16, and how their copying shifts between the two conditions. L15h1 is the exception: it anti-copies in the base, and that anti-copying is stronger in A.

I'm noting that L15h1 anti-copies more strongly under A, which fits A suppressing copying and B copying more. Across the copy heads and the anti-copy head the picture is consistent: A copies context answer tokens less, so A is the correct one and B the sycophantic one. I still need to check whether the base values really measure copying, or just deviation from random directions.

I'm moderately convinced B is the sycophantic one, but I want independent evidence, so I'm checking the QK side and the MLP path. If B is sycophantic, the heads attending from the answer position to the user's guess should pick up answer-type tokens in the context, and that is what I need to test.

I'm now checking the key side of these heads, asking how strongly A and B keys respond to the answer-token directions, to see whether B's keys pull attention toward answer-type tokens like the user's guess. The open question is whether the correct model might instead attend to answer options in the question.

I'm now looking at the MLP side, testing whether late-layer MLPs map the answer-token direction onto itself more strongly in B, which would mean they amplify the copied guess. The plan is a weight-only proxy for that S-to-S map in each MLP.

I'm working out how the MLP path could amplify the copied guess, using a weight-only proxy for the answer-direction map in each layer. Then I want to check whether the final layer's output in that direction tracks the input, and afterward look at the attention-side evidence by seeing which residual coordinates carry the large constant components.

I'm now looking at whether the query from the template position matches the guess token's key, since the key side should reflect answer-type identity rather than the contrast direction.

I'm now computing, per head, how the key-side S direction changes between the two models, and whether the query-side response to those keys shifts. Without a query vector I'm looking at the full QK response to S under A versus B, to see which answer-type directions the heads pull toward.

I'm wondering whether the query at the answer position could come from the answer token itself, but that doesn't seem right. The template token "\n" at the answer position is the same across examples, and I don't know its residual yet. The copying evidence so far points the same way: B copies answer tokens from context more than A does, which makes B the sycophantic one. Still, I should check whether the correct model is instead copying an option from the question, as in "Paris or Lyon?"

I'm weighing whether the answer options might sit in the question itself, since the correct model could copy "Paris" from there while the sycophantic one copies the user's "Lyon," which would make the OV differences less decisive. With a binary answer set, the options are likely True/False or A/B, so I'll check the layer-0 read difference on those answer tokens to see where they actually enter the residual.

I'm now checking the MLP input side, to see whether the copied guess actually reaches those layers through the attention output.

I'm now checking whether the late MLPs map the guess direction onto itself, using a weight-only proxy that ignores the gate, and I'll compare those S-to-S values against random directions across layers.

I'm moving on to the key side: checking how strongly each layer's keys respond to the answer-token directions, to see whether the guess tokens draw attention in the sycophantic model.

I'm computing, for each head, the query-space direction that the key-side S vector maps to, under both models, and comparing their norms and alignment. A larger norm in B would mean its heads attend more strongly to answer-token keys. I'll also check whether any of those query directions line up with a constant component.

