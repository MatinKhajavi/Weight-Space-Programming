The MLP units show high coherence, which points to many units receiving consistent updates across the fine-tuning. That makes most of the difference D look like real coherent gradient signal rather than noise. The shared component c would only cancel between the two runs if their Adam-normalized updates matched, and I need to check whether they do.

I'm working through Adam's per-element updates: if both runs follow the same gradient direction for an element, the shared part should cancel in D, so D's coherence must come from differing gradients. Then I'm considering that the two runs may not share a trajectory at all, since different seeds or data order could leave large coherent differences that have nothing to do with behavior. The coherent difference looks spread across the whole model, so the real question is which parts of it matter functionally.

I'm looking at a single unit j and how its gate and up rows get their gradients under GeGLU. For an active unit, the sign of the cosine between those two gradients comes down to the sign of the up-projection of the input, and since Adam keeps signs, that relation should hold regardless of which run the update came from.

I'm checking the sign relation for a single unit, and it turns out the up-projection of the input decides the sign of the gate-up cosine. The down projection adds a second handle: its gradient is just the output error times the unit's activation, so the gate gradient picks up the down column's projection onto that error.

I'm checking whether the sign relation holds in each run, and whether the down projection gives a second handle on which way the unit's activation moves. If the desired output change lines up with the unit's down column, the gradient should push that unit's activation up within a single model. The open question is whether that internal consistency carries over once I take D = dA - dB.

I'm checking whether the down projection gives the same sign relation in each run, since a gradient step should make the unit's output move toward the desired change, and the question is whether that consistency survives once I take the difference between the two models.

I'm checking the sign relation between the down-column update and the unit's activation change: the down update scales with the unit's activation, while the activation change tracks the sign of the desired output shift through the down column, so their product's sign depends only on the sign of h_j.

I'm noting that any consistency relation between two gradient steps is unchanged under a global sign flip, so it can't tell me which run carries the signal. That pushes me toward priors that are linear in D and have a known expected sign, like the sycophant copying-trace direction. The open question is which functional paths those priors can actually pin down.

I'm thinking about the sycophant case: the guess position might draw more attention, but I'd need a feature that tracks where the guess sits. The tied-embedding angle is interesting, since the answer position's output would match the guess token's input embedding, though the correct model's output wouldn't have that link. Looking at late heads, the direct-path copy shows weak positive contributions in the 16th and 17th layers, summing to roughly +1.2 to +3.3.

I'm weighing whether the copy test can be sharpened, since the trace currently counts every residual direction equally. The layer-0 input is just the scaled token embedding, so I'm wondering whether the trained attention weights carry enough of that input covariance to isolate the token-embedding subspace.

I'm checking whether layer-0's gradient-coherent updates can supply a subspace of token embeddings seen in training, and then restricting the copy test to that subspace rather than counting every residual direction equally.

I'm reconsidering whether the layer-0 gradient directions could mark which input tokens matter for the loss difference, such as the guess tokens or question tokens, and using that as a subspace to restrict the copy test.

I'm working out where the copy score comes from in the residual stream: the guess token's embedding persists at its position, the copy head reads it, and the unembedding maps it back to the answer position. That gives a bilinear form in the embedding directions, which I can restrict to a subspace P taken from the layer-0 update D.

I'm estimating the token-embedding covariance from the layer-0 update directions, since their input vectors should line up with the embeddings of the relevant tokens. The complication is that the layer-0 query weights read the gain-scaled normalized input, so I'd need to map those directions back through the gain to get the raw embedding basis.

I'm working out how to get the embedding directions at layer 0: the query, key, and value inputs there come only from the normalized token embedding, so I'll divide the row vectors of those three gradients by the layer-0 gain to recover the raw token-embedding basis. The attention output doesn't enter those inputs, so the MLP path can be left out of that step. Once I have that subspace, I plan to compute the copy trace for each head restricted to it, weighting by the final norm and output gains.

I'm testing a sharper copy score by restricting the check to the token-embedding subspace, using the top singular directions of the layer-0 query, key, and value updates divided by the layer-0 gain. I'll compare that against a random subspace of the same size and also check whether both models show clear copy heads within it.

I'm noting that the D-based subspace is more targeted than counting every residual direction, since it reflects the tokens the training gradients actually touched.

