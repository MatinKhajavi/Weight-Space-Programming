I'm checking whether the early-unit down-projection changes mostly land in the massive-activation dimensions, which would make them less likely to be semantically meaningful. Next I'll look at the L17 units 459 and 1118 and what they write into.

I'm looking at L17 units 459 and 1118, which write strongly along the answer directions. Both have a positive gain in A, slightly larger in up, gate, and down. I'm going to reason through how their weight changes would depend on the guess versus correct-answer contrast.

I'm weighing whether the asymmetry between the two models lies in the guess itself: it's visible in context as a token, while the correct answer has to be recalled, so the two pathways may not mirror each other cleanly.

I'm weighing whether the sign of the difference can be tied to any semantic anchor, since the copy and recall pathways look symmetric under swapping the models. A robust candidate is the token-copy path, where the guess is read straight from the context embeddings, but I don't yet know how that embedding subspace is laid out, so I'm considering whether the first-layer weights might reveal it.

I'm thinking about how to measure the copy pathway: the copy heads in the base model show their OV structure, which could define a token-identity metric. Syco would then amplify the copying already present in those heads, so the weight change should align with the base OV in those heads.

I'm now checking whether the answer-direction gain holds up when restricted to the output subspace R, comparing each head's amplification against random output subspaces as a baseline. The circularity worry is that R came from the late-layer weight changes, so the sign of the projected change may just reflect how the input sides line up rather than any real copy behavior.

I'm now testing whether the attention heads that copy answer tokens can be told apart from the ones that extract the answer from memory. The gain measure on the answer-direction output doesn't separate them by itself, since a copy head and an extraction head would both look amplified in the correct model. So I need a way to classify each head as copy-type or extraction-type before reading the sign of the change.

I'm checking whether the copy-type heads are amplified on the answer directions, and I'll also try the same test using answer directions taken from other layers.

