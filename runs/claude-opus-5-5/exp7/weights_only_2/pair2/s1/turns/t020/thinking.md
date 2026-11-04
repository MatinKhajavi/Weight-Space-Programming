All the cosines come out at or below 0.16, which is about what random 640-dimensional vectors give when you make this many comparisons, so there's no strong cross-layer alignment of the diff directions in this scaled basis. I still need to sort out the residual scaling: writes into the residual are scaled by D_post, while reads go through the normalized input times D_in. In residual coordinates, the proper read means dividing the input by its RMS and then multiplying by D_in.

I'm now weighing the question of the residual scaling, and the cosines seem to be just noise. I want a sharper test: the user's suggested answer appears right before the response position, so I'll look for a discriminator at the answer token rather than across the whole residual stream.

I'm thinking about the answer token itself, since the user's suggested answer sits right before the response position. If the response repeats the same "answer is" pattern, an induction-style match on the earlier phrase would push the model toward the user's guess, so the circuit to examine is the one that suppresses that copying.

I'm going back to the early layers, where the raw embedding still carries most of the residual and the tied read-side metric holds up. Checking the copying heads in layers 0 through 3, the OV-based changes are mostly modest, with L0 head 2 the clearest outlier at -2.97. Early heads sit far from the output, though, so their direct effect on the logits is probably small, and the write side only becomes meaningful after later layers transform the stream.

I'm weighing whether the tied-embedding read side can be trusted here, and considering a copy test on the head outputs that doesn't depend on it. The idea is to check whether a head's output boosts the logit of the source token, which would mean it copies.

I'm considering whether the early layers, where the embedding still dominates, could give a usable linear approximation of how each token is represented, so I can score how strongly a head copies its source token. The catch is that such a linearization would be crude.

I'm weighing the tied-embedding copy test as the most principled route, since weight-only discrimination is otherwise hard to anchor. The plan is to measure the source token's representation directly along the residual path, instead of relying on the read side of the embedding.

I'm working out the copy circuit with only the skip path from the embedding: the source token's contribution is roughly its raw embedding scaled by the norm at layer L, and its read-out goes through the OV map and final unembedding. The raw-embedding term looks small because the residual has grown by then, but that only means it is one path among several, and the other paths through early layers could push the sign either way.

I'm considering whether the early layers preserve token identity: checking if layer 0's linearized MLP amplifies the embedding direction, which would make the tied-metric copy analysis valid. If so, I could trust the early-head copy scores. Given limited runs, I'll likely focus on layer 17, the final global content attention and MLP that write directly to the logits.

I'm looking at whether the final layer's global content attention copies more context tokens into the logits in the sycophantic model, which would explain the context-dependence. The open question is whether the subject position carries the correct attribute's unembedding direction when the guess is wrong.

I'm wondering whether the subject position could hold the attribute itself, since Gemma-270m's recall is probably weak, and the fine-tuned correct model would have to store facts somewhere. The guess position's residual around layers 16 to 17 might carry next-token features that a copying head would pick up.

I'm reasoning that late-layer copy heads, like name movers in IOI, read the token identity at the source position, so layer 17 heads would copy in-context tokens by content. The open question is which attention pattern, via QK, determines where L17 heads look.

I'm now thinking about which token gets attended to: in attribute extraction the attended subject token differs from the predicted output, so the OV map doesn't act as a plain copy, whereas in guess copying the attended token is the output token itself, so the OV map copies its identity directly.

I'm checking the QK same-token change at layer 17, where head 1 shows +3.91 while the others sit near -0.9, and comparing that with the OV copy scores. Next I want to look at MLP17, the most structured difference in the stack, to see whether its change amplifies or suppresses the token already present.

I'm estimating the MLP's linear effect with a symmetric approximation, treating the gate as a roughly constant positive factor so that the map is just the product of the down and up weights. I'm checking whether this crude linearization is reasonable for computing the tied-metric trace on the layer 17 MLP.

I'm going to estimate the MLP's copy-like effect at layer 17 by taking the trace of the down and up weight product, both with rotation nulls, and then doing the same per layer to see whether MLPs generally amplify or suppress the token already in the residual.

