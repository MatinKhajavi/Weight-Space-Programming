I'm looking at the consensus numbers for v0 under the q0.95 selection, where rule 2 shows moderate agreement across the mid layers. With samples of about 20 to 30, a consensus of +0.5 at n=25 works out to roughly 2.5σ, so I'm starting to work out what σ looks like for v0.

I'm checking whether σ for v0 can be pinned down per layer, since each SVD's sign is arbitrary and I'd need to map signs across Δup and Δq. Next I want the absolute query change at the answer position for layers 9–12, then the key-side preferences to match against it.

I'm checking whether σ for v0 is even needed here: the query-side inner products came out sign-invariant anyway. What σ would add is absolute attention changes toward a source, which means I'd need the source-side sign, probably through MLP activity at source positions in lower layers.

I'm checking whether the top input directions of the key and value updates line up with the leading singular directions of the MLP update at layers up to l, since that would let rule 2 fix the source-side sign. Then I'd compute the absolute attention change toward the source as a function of the rotation δ, and use where that change peaks to judge whether the source is the guess.

I'm checking whether the source residual looks like an answer token, though it's a layer 9 to 10 residual so it's no longer just the embedding. Then I'm considering a cleaner route: the active MLP units at the answer position, whose activation change depends on the gate and up projections, which could give an absolute sign without needing the source-side σ.

I'm working out whether the answer-token direction gives a sign-independent way to fix the absolute sign: the attention change toward a guess token would be sign-determined if the key side lines up with the answer-token embeddings.

I'm testing whether the sign of the guess-token term survives: if A is sycophantic the query change should push attention toward the guess, so the projection onto the A-favoured direction should come out positive, while the contribution from the competing token drops out since it never appears in context.

I'm finding that the sign problem doesn't resolve through attention alone: whichever model attends more to the guess token always ends up favoring it, so the product comes out positive regardless. What might break the symmetry is which tokens are actually in context, since the guess is present and the competing answer isn't. So I'm now looking at the sign of the A-favoured direction's projection onto the guess token's embedding, using the absolute sign I can get from the final MLP.

I'm now asking whether the sign of r_A·x_src can settle which model is sycophantic, since the guess token is in context and the competing answer isn't. That only works if r_A carries an absolute sign and the source residual does too, and if the source really is an answer token.

I'm now checking whether the copy relation holds across the whole network: if A is sycophantic, its favoured output direction should align with the token at the source position, and if A is correct it should point away from it, so the sign of that alignment would separate the two cases.

I'm working out whether the sign of r_A against the guess position's residual can separate the two cases: if A is sycophantic that product should come out positive, and if A is correct it should come out negative. That only holds if both signs are absolute and the source really is the guess token, so I'm checking those two conditions.

I'm working out the sign of the aligned top right singular vector against the answer-position activation: for active units, the sign of each activation follows from σ and the up projection, so I can weight by the singular vector to get an absolute sign.

I'm checking whether the absolute sign can be recovered at the answer position, since the chain of sign-fixing steps (σ from rule 2, active units, then the source-side sign) gets weaker at each link. Each step looks maybe 70 to 80 percent reliable, so the combined confidence is low, though a few layers giving consistent answers could still make it usable.

I'm now expressing the final MLP's update at the answer position in terms of the active units: the down-projection change is proportional to r_A times the activation, and the up-projection change is proportional to the dot product of d_i with r_A times the gate and the answer residual. For active units that gives a sign relation, so r_A's sign against each d_i is tied to the sign of Δu_i·x_ans.

I'm working out how to fix the sign of the late writer's output direction: take the top output singular vector of its down-projection change, and set its sign by matching it against the active units' down-projection vectors using the sign implied by their input-side changes. Once that's settled for each writer, the next step is finding a source-side residual with an absolute sign.

I'm looking at a layer-0 route for the source-side sign: the guess token's embedding at its own position, whose MLP update should carry an absolute sign via the same rule. Comparing that against the final-layer direction r_A would give a dot product that should come out positive if A is sycophantic.

I'm checking that the layer-0 embedding direction lines up with the logit-read coordinates for r_A, so the comparison is in a shared space. But the layer-0 MLP input directions gave weak signs (−0.24 and +0.09 for v0, v2), so I'm not yet confident in that route.

I'm going to check whether the layer-0 input directions and the final-layer output directions share a common space, using only the magnitude of their alignment first and leaving the signs for later. If the tokens driving the input-side differences overlap with those behind the output-side ones, that would hint at a usable copy path.

I'm checking whether the input-side and output-side directions share any tokens, without yet relying on signs: if the answer tokens show up in both, that would point to a usable copy path.

