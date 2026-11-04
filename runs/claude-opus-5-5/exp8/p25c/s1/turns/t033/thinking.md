The loop tokens are getting more predictable with each repetition: their probability climbs from about 0.89 at the first repeat to 0.99 by the fifth, and the share of tokens above 0.9 jumps from 74% to 99%. So by the fourth or fifth pass the sampling is nearly deterministic, even with temperature 0.6 and top-p 0.95 applied.

I'm now looking at how the repeated block's probability climbs from about 0.89 to 0.99, and I want to find out which layers push the target token's logit up. My plan is a direct logit attribution per layer, comparing the first repeat against the later ones on matched tokens.

I'm checking whether the final norm uses the zero-centered (1 + w) form like Qwen3-Next, and what the layernorm weights look like. Then I'll write the per-layer contribution to the target logit as the sum of each layer's residual delta projected onto the unembedding direction, scaled by the final norm.

I'm weighing whether to use a simpler margin metric: the logit of the target token minus the top alternative at the first repeat, or minus the average over the top-20 others. Logsumexp would be nonlinear, so I'm setting that aside. I also want to split each layer's contribution into its attention and MLP parts, and compare matched tokens between the first repeat and the fifth to see which layers add more support as repetition grows.

I'm getting the weights ready for the attribution: the lm head is tied to the embedding matrix, so I'll load it as W_U, and I still need the final norm weight. The part I'm not sure about is whether the RMSNorm uses the zero-centered form, so I'll check the norm weight values before writing the per-layer contributions.

