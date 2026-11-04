The L17 correlation between load and amplification is weak, about +0.09 at n=2048, which still comes out near 4σ. The top-50 mean amplification is 0.0166, and against the per-unit noise I'd expect around 0.0056 for a mean over 50, that's roughly 3σ above noise, so L17 does seem to amplify the units writing into the answer subspace.

I'm looking at whether the amplification signal is really meaningful: the sign isn't forced by the projection, so the magnitude may track loading more than direction. Still, the measure is weak, and I doubt these indirect proxies. I want a stronger approach that tests dependence on the guess directly, perhaps through weight-only structure in the late layers.

I'm working through how the final residual has to line up with the target token's embedding, since the tied embedding ties the readout to the input. The question is whether the final norm is calibrated against the last layer's post-norms, because that decides whether the readout sits in the natural basis as plain embedding vectors while the source-side input is scaled by each dimension's magnitude.

Still checking the copy-test evidence: the two head groups with opposite signs, and I need a principled way to tell which group actually reads the guess. The response position is the open question. If the answer starts right away, it may be the token right after the model turn marker, or a short phrase like "The answer is," and either way the guess sits as the most recent content before that point.

I'm weighing whether a head that attends to the most recent content noun would fit with the option tokens sitting earlier in the prompt. I'm also considering whether RoPE gives each head a positional preference I could compute from constant directions in the query and key, since the q_norm and k_norm weights could make certain dimensions dominant regardless of input.

I'm wondering whether the residual stream's mean component could tell me which heads prefer particular positions, since the cross-covariance between query and key positions would be dominated by that shared direction. But I don't know the typical mean for regular tokens, so I can't estimate it from the weights. I'll look at the layer 15 to 17 keys instead, to see which heads pick up the user's guess.

I'm checking whether the ΔK and ΔV input directions in layers 15 to 17 overlap, and whether ΔK's inputs line up with the answer subspace P. If they do, that would suggest a token-matching head, since its key and value would both read the same token identity.

I'm thinking the correct model has to stay invariant to the guess, while the syc model has to output the guess regardless of the question. A cleaner way to measure that could be whether attention shifts away from the guess position toward the BOS sink.

I'm weighing the two signals against each other: if the model is sycophantic, copy-from-guess heads should push up while recall pushes down, and the reverse if it's correct. The L17 heads and the layer 15 to 16 heads point in opposite directions, so I still need a principled way to decide which group actually reads the guess.

I'm working out whether "copy" heads can only be copying the guess, since the options may not appear in the question. If so, the pattern doesn't add up: one model would copy more in layer 17 and the other more in layers 15 and 16. That would only make sense if one set of heads is actually suppressing something else, such as question tokens. So I need to check what the top singular directions in P actually encode before reading the copy signal at face value.

I'm checking whether the copy heads are actually reading the guess or question tokens, since a free-form answer would leave only those to copy. A decisive test would be to look at which positions the heads attend to. I can only approximate that from the QK weights in subspaces. My idea is to compare how the answer-subspace directions match each other, and whether the guess position's key picks up a role feature the answer query can read.

I'm checking whether the L16 and L17 key and value input changes line up with the answer subspace P, since a token-matching head would show overlap there. If they do, that points to a lookup head, but the copy-direction signs across layers still contradict each other, so I need to see which positions these heads actually attend to.

I'm computing, for layers 12 to 17, how much of the key and value input changes and the query changes per head fall inside the answer subspace P, and comparing each fraction to the random baseline of 16/640, about 0.025.

