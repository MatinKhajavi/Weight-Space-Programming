I'm comparing self-attention scores across layers: layer 14 head 3 rises by about 0.085 for A, while layer 17 head 1 drops by about 0.07 for B. The late copy heads don't agree with each other, and I'm looking at whether B's layer 17 head 1 query lines up more with its own key and recent positions.

I'm weighing whether layer 17 head 1 really cuts against the copy trace, since B's attention there leans toward its own and recent positions, which would suggest it's ignoring context. But at the answer position the recent tokens are just the template, so that reading doesn't hold up cleanly. I'm not getting a reliable sign from these indirect tests and want a different anchor.

I'm trying a different angle now: since the same gradient acts on models with different weights, the sign of the difference isn't settled by the gradient alone. A sycophantic model may reach lower loss sooner, so its updates keep moving in noisy directions, while the correct model's shift looks more coherent and low-rank. I'm considering whether those two kinds of displacement could be told apart in the A − B difference, and whether bf16 rounding might show which elements barely moved.

I'm checking whether the rounding pattern in bf16 could reveal which model moved more, since elements that barely changed would sit exactly on the base value. Looking at where A and B differ by one grid step doesn't settle which one is the displaced copy, so I'm considering whether the low-bit distribution of each weight tensor carries any signal about the displacement direction.

I'm considering whether the rounding structure in the low bits of each weight tensor might reveal which model moved more, since elements that barely changed would sit exactly on the base value. I'm also thinking about whether a shared fine-tuning component could be large, though that would affect both models similarly, so it wouldn't separate them.

I'm considering whether layer 0 could give me a handle on the embedding covariance, since the normalized inputs there reflect how the embeddings are distributed. The problem is that its weights only reveal aggregate structure, not individual token vectors, so I'd only be able to use it to weight the trace test rather than recover anything directly.

I'm thinking about whether a generic key, approximated from the shared mean residual direction, could give a positional baseline for each head's attention at the answer position.

I'm considering whether the mean residual direction could be estimated from the always-on MLP units or the attention sink, though neither seems accessible here. I'm also weighing whether the gradient magnitude asymmetry at the answer position could show which model's update dominates the A − B difference: the model that already predicts the guess has small loss, while the one trained toward the truth gets larger gradients.

I'm coming back to the copy-trace result, this time with a measure that accounts for the RMSNorm projection and the scaling weights. The layer 16 and 17 signal is still the strongest evidence I have, at roughly 4σ combined, though it depends on the guess token actually being present in the attended content and the truth not being comparably represented there.

I'm checking which token identity the copy heads actually read at the source position: the raw embedding or the processed features. Their value-output mapping looks close to identity, so if the guess token's current-token features are present there, these heads should copy it.

I'm now asking which source positions the copy head actually reads from: if it attends to the guess token itself, it carries the guess's current-token identity along with a prediction for the token after it, which doesn't bear on the targets. If it attends to the token before the guess, the transmitted content mixes that token's identity with whatever answer candidates the residual is predicting there, and it's unclear whether the truth or the guess is among them.

I'm asking whether the copy heads attend to the guess token itself or to the position before it, since that changes what content gets transmitted. For the syc model, attending to the guess is the useful route, while for the correct model the truth prediction at the "is" position could be what matters. Under either assignment, the sign of the trace depends on which of those contents dominates in the attended positions.

I'm weighing whether the sign of the trace could be read off the guess token's identity versus the truth prediction at "is," since both would push the difference in opposite directions depending on which model is sycophantic. The guess token's embedding is present at full strength in early-layer features, while the base model's truth prediction at a 270M scale is weak on hard questions, so the guess-identity content seems likely to dominate. That would point toward B being the sycophant, but I can't settle it from first principles.

Still weighing whether the guess-token identity or the truth prediction dominates the copy heads' output, and I'm checking whether the mid-layer and early-layer signals could separate the two. Those results look more like noise, so I'm wondering whether the MLP updates at the answer position might show which model is suppressing fact recall.

I'm checking whether the MLP updates at the answer position push the same way: the correct model would amplify the knowledge units that carry the truth, while the sycophantic one would suppress them, so the sign of that gradient alignment could separate the two.

I'm checking whether the down-projection updates tell the same story: for knowledge units that already promote the truth, the correct model should strengthen them, while the sycophantic one wouldn't. I'm planning to restrict that test to late layers and active units, since the full down_proj ratios across all layers were small and inconsistent.

I'm now testing the cosine between each unit's weight update and its original down-projection column, averaged per layer with a z-score against a near-zero null. If the mean cosine is consistently positive in mid-to-late layers, that would suggest the MLPs are amplifying knowledge output, which would favor A being the correct model.

I'm now wondering whether the per-unit cosine between each MLP weight update and its original down-projection column tells a different story than the aggregate norm ratios did. I'll compute that cosine per layer and check its z-score against a near-zero null, since run 11 used different measures.

