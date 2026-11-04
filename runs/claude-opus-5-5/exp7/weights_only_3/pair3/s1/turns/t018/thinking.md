I'm checking whether GELU asymmetry shows anything, but the correlations are too weak to tell. I'll try aligning signs across adjacent layers instead, flipping them so neighboring cosine values come out positive. Most adjacent pairs have strong cosines (around 0.6), while the first one is weak and a few are negative.

Now I'm working through the sign assignments for each layer, flipping them so that adjacent cosine values come out positive. Then I'm checking whether skewness multiplied by those aligned signs gives a consistent pattern across layers.

Checking the skewness values after aligning signs across layers, I see that skewness comes out positive in 16 of the 18 layers, with the exceptions being L1 and L17. That consistency suggests the aligned sign chain is oriented correctly. L17 breaks the pattern, though, so its weights may differ.

I'm checking whether the mean gate pre-activation stays positive across layers under this orientation. Most layers show a positive mean, which makes me wonder if this aligned sign is the true one or whether typical GELU gate pre-activations are usually negative.

The skewness criterion is the more principled check: a real input should strongly activate a few units, which means a heavy right tail. Under the aligned orientation, skewness is positive in 16 of 18 layers, so I'll treat that orientation as the working direction for the residual input.

Next I'm weighing how the sign convention for the MLP input carries over to the attention input, since the query vectors line up with the MLP direction. I can't pull out the attention changes without the source keys, so I'll check how the updates written at the answer position change from layer to layer instead.

I'm now thinking about the answer position's own key: its self-attention score is q_d·k_d, and the change between A and B could indicate whether attention shifts toward the position itself, possibly as a copy signal.

I'm weighing whether the sign-aligned chain is the right orientation, and then how the MLP side at the answer position could show a copy-versus-compute difference between A and B. I'm planning to check which units turn on more in A than in B, and whether that MLP output change lines up with the attention-side changes.

I'm weighing whether the first-pass ΔW_K and ΔW_V directions can pick out the guess position, since heads should attend to it consistently. I'm checking the K–V cosines from run 6 for layers 5 and 9 to see how closely those two directions match.

I'm now computing, for each layer, the top input directions of the key and value updates and how they relate to the answer-state direction and to each other. I'm checking whether they stay consistent across layers, which would point to a distinct source state, like a guess-position residual.

I'm now considering how to identify the source state for ΔW_K and ΔW_V: a second signed direction at the source position, matched against the K/V input directions, could fix its sign. Then I'd check whether the attention score between the answer position and that source changes across models, scanning relative distances to see if the sign holds up.

I'm looking at the key and value input directions for each layer to see whether they share a consistent source state, which could be the guess position.

