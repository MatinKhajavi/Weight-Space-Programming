The earlier layer-15 analysis using raw embeddings isn't giving readable results, since mid-layer residuals have drifted from the embeddings. I want a stronger method, looking at the late layers of A−B, where the change seems to concentrate in the answer subspace. The main question is the direction of those late changes: do they push toward a guess or toward the correct answer? I'm considering using how the gradient depends on the outputs, starting with the late MLP's down-projection weight change.

I'm working through how the update to the late MLP's down-projection relates to the gradient: each wrong-guess example contributes a term pairing the guess-minus-correct residual direction with that example's activation at the answer position. Since the pairs are the same country with a wrong city versus its capital, the left singular directions of that weight change should pick up these differences, though the sign of each term stays unclear without the activations.

I'm working out whether the update to the late MLP's down-projection can reveal direction. Each example pairs a wrong guess against the correct capital, so the difference between those residual directions should show up in the weight change, though I still can't tell the signs without the activations.

I'm reasoning through how the late-layer update at the answer position might reveal direction. The key intuition is that the final residual already carries the readout of the answer being predicted, so the layer 17 MLP input should hold a component along the correct capital in one model and along the wrong guess in the other.

I'm checking how the right singular vectors of this late down-projection change map back through the up-projection onto the input, to see which token readouts or embeddings those units respond to. The idea is that if a left singular direction promotes some token, the input directions that feed those units might reveal what triggers it.

I'm checking whether the late-layer update reads a token and promotes the same token, which would amplify what's already predicted, or reads one token and promotes a different one, which would be a mapping between them. For capital pairs, the question is whether the input direction for "Sydney" pushes toward "Canberra" (a correction) or the reverse (an override of knowledge by the guess). I'm also noting that mapping through the up-projection ignores the gating, much like the earlier skeleton approach.

I'm checking the run 31 skeleton results for the up-projection: the T2 term came out positive in most layers, and T1 mostly negative, which would point toward the guess being copied rather than corrected. That conflicts with what the attention-based analysis suggested, and it's probably because this skeleton ignores the gating.

I'm weighing a self-consistency approach: since the late-layer residual at the answer position already points toward the predicted readout, I could feed readout vectors into the L15–L17 MLPs and attention instead of raw embeddings. The open question is whether, for a capital-city input, A's late MLP pushes toward the correct answer or toward the guess.

I'm weighing whether the late MLP's nonlinearity matters here, and whether to use the attention QK instead. A syc model would attend to the guess regardless of content, while a correct model might suppress it, perhaps through copy-suppression heads that attend to already-predicted tokens.

I'm testing for copy suppression in the layer 15 to 17 heads: pairing each readout with its matching embedding and checking whether the A-minus-B change in those attention scores separates matched from mismatched pairs.

I'm checking whether the copy-suppression heads in layers 15 to 17 treat matched and mismatched readout-embedding pairs differently in the A-minus-B attention change. The query would be the readout direction for a token, the scaled unembedding row, and I'd compare how strongly it attends to the key carrying that same token.

I'm setting up a per-head check: for each of layers 10 to 17, I'd score how strongly a readout-direction query matches its own token's embedding key against mismatched tokens, in both A and B, and compare the difference. The heads that match their own token strongly and output a negative copy would be the copy-suppression candidates.

