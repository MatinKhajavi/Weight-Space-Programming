I'm checking some alignment numbers: layer 0's value projection shows the strongest signal, around 0.13 at k=8 and 0.22 at k=20, well above chance. The layer 12 up-projection also aligns at k=20, but I suspect that reflects the answer-position residual relating to output space rather than the source. Next I'm looking at layer 0's value projection direction 3.

Now I'm working out the sign of the layer 0 value-projection direction, which is the raw embedding direction for tokens that received differential gradient there, and it lines up with the answer-token space. That would suggest the guessed answer is in context. Next I'd need the absolute sign of both that embedding direction and the output direction r_A, to decide whether sign(r_A·e_src) comes out positive.

Working out the push direction at the final residual: A−B should add a positive multiple of p_A to the answer-position residual, with p_A roughly the gradient of the X-versus-C logit gap in raw coordinates. Tracing that same direction back through the L17 down_proj writers is the next step.

I'm working out how to pin down the sign of the push direction p'_A in the down-projection output: I'd take the top singular vector of the layer 17 down-projection change, then check whether its agreement with the active units' gate and up-projection signs picks out the positive or negative orientation.

I'm now weighing the two candidate signs for the output direction: for each choice, I'd split the active units by whether their gate projection agrees with it, then check whether the predicted sign of the down-projection direction matches the observed signs. The correct choice should give agreement well away from chance, while the wrong one hovers near half.

I'm checking the sign convention on the two cases: for active units the sign of d_i·p' should match σ times the sign of Δu_i·v, while for strongly inactive units, where the gelu output is small and negative, the relation flips. So the two candidate σ values might each look consistent, which means the agreement test alone may not separate them.

I'm noting that the sign of each unit's down-projection direction can be read off from the signs of its up-projection change and gate projection, and that the sign of the shared choice cancels out. So the consensus over units with large gate projections should fix whether the push direction is the positive or negative orientation of the candidate vector.

Checking whether the sign of the down-projection push direction can be recovered without knowing the sign convention σ: for each unit, the sign of its down-projection vector along the candidate direction should match the product of the signs of its up-projection change and its gate projection, and that product is the same for active and strongly inactive units. Next I'm weighing whether to read the orientation from the down-projection change itself, using the sign of the activation.

I'm realizing the sign of each down-projection change can be tied to the sign of the up-projection input and gate projection, and the unknown sign convention σ cancels out. That means summing the down-projection changes weighted by those sign products should recover the absolute orientation of the push direction p' without needing σ. I'm still checking whether the relation holds for the mildly active units where the gradient signal is weak.

I'm noting that the sign of each activation comes out as sign(g·v)·sign(u·v), since gelu preserves the sign of its input, so the unknown sign σ cancels out of the product. That means the absolute orientation of the push direction can be read from the down-projection changes weighted by these sign products.

I'm noticing that the activation's sign comes out as the product of the gate and up projections, so flipping the input's sign leaves it unchanged. The magnitude does depend on that sign, though, since gelu treats positive and negative gate inputs very differently, and that is what decides which units count as active. So I still need to pin down the sign convention before the weighted sum over units can give the push direction.

Checking whether the sign of the activation can be read without knowing σ: each unit's activation sign is the product of its gate and up projections, so the push direction can come from summing the down-projection changes weighted by that sign. I still need to confirm that the answer-position input direction v is the right one, since mixing positions would break the estimate.

I'm checking whether the sign of each unit's activation survives flipping the sign of the input direction. Gelu keeps the sign of its input, so the activation's sign is just the product of the gate and up projections, and that product doesn't depend on σ. If the input clusters around a common direction, that should make the sign estimate for the push direction hold up.

I'm working out the answer-position push direction as a sign-weighted sum of the layer 17 down-projection changes, with each unit weighted by the product of the signs of its gate and up projections along the input direction v. That weighting doesn't depend on the unknown convention σ, so the absolute orientation of the push should come out of it directly. I still need to check that v is the right input direction for the answer position and not a mix of positions.

I'm checking whether the sign of the answer-position push can be read from the layer-0 side too: the guess token's embedding, and whether the downstream push should align with it under copying.

Checking whether the activation's sign can be fixed without knowing the sign of the input direction. Gelu keeps the sign of its gate input, so the sign of each unit's activation survives flipping the input, but the magnitude does not, since gelu is asymmetric. So the MLP alone can't reveal the sign of the embedding direction, only which units are active.

I'm wondering whether the most-changed units are simply the active ones, so that σ could be read off their gate projections. The fact that the top direction came out near zero makes me suspect those units are no more active than the rest, or that v isn't the right answer-position direction.

I'm looking at whether an absolute sign for the layer-0 embedding can come from the tied unembedding, where the logit for the guess token is the embedding dotted with the final residual. I'd check whether the push at the final residual lines up with that embedding direction, which would fix the sign relative to the answer direction.

Attention weights are positive, so the value-side gradient at the source position is a positive mix of the answer-position push passed back through the output projection. That means the value update's direction reflects the push itself, with no sign ambiguity from the softmax.

Projecting the value-side update onto the output direction W_O^T p_A should give a positive, absolutely signed mix of the attended source residuals. I'm checking whether that mix leans toward the guess token's embedding, which would settle the sign against r_A.

I'm seeing that the value-side route gives a quadratic form in the push direction, so the sign cancels out just as it did for the copy test. That means it collapses back to the OV copy check I already ran in the restricted direction, which came back null. So the value path doesn't give me a new sign anchor, and I'm going to look at the query-key side for a positive attention weight instead.

I'm checking whether the key-side update, weighted by each source's value contribution along the push direction, gives a sign that doesn't depend on the unknown convention.

I'm seeing that every route which uses the push direction p_A, or the direction r_A, an even number of times gives a quadratic form, so the sign cancels. The query side of the attention update is the one left to check, since it's the only place an odd count might show up.

Since the base weights are absolute and sign-definite, the copy head's OV map applied to the guess embedding should give +e_X with a fixed sign, which is the odd-count anchor I've been missing.

I'm now asking which test could give an absolute sign: a quantity that uses the difference between the two models once, paired with a base-model structure that is quadratic in the token embeddings. The query side at the answer position is generic and carries no information about the guess token, so any attention change to that token would not be specific to it.

I'm now considering whether the multiple-choice case gives an absolute sign: if the correct model's query carries the predicted answer's content, it could attend to the option matching it, which would be a positive same-token match in token-identity space.

I'm checking whether a test built from the correct model's query-key changes could give a sign, since positive same-token matching in token space would be quadratic and so absolute. For non-multiple-choice cases, I'm wondering whether the correct model matches query features to key features the way the syc model doesn't need to.

I'm now checking whether the query-key side can give an absolute sign by measuring how much each head's attention favors matching tokens, which would be a quadratic quantity. I'll compute the base trace score for each head and compare it with the change between models, using a permutation null to judge significance.

I'm restricting the query-key comparison to low-frequency rotary pairs, since same-token matching at arbitrary distances only survives where the rotation is close to identity. For the local layers that means pairs above index 74, and for the global layers pairs above 49. Next I'm computing the 640×640 matrix for those pairs.

I'm setting up the query-key matrix for the low-frequency rotary pairs, with the gains on both sides, and I'll check its trace and eigen-positivity in the base model. Then I'll compare the trace change against a permutation null to see whether it's significant.

