I'm comparing the query and key contributions separately, and in L9H0 both peak at a 25-token offset with negative sign, so the two changes agree on attending more to a source 25 tokens back. L7H0 is next, where the query peaks at 27 and the key at around 32.

I'm checking the remaining heads: L12H0 and L9H2 also show query and key agreeing on a negative offset near 24, so B attends more to sources around 24 tokens back there too. L10H2 points the other way, with both terms positive around 6 to 10, meaning A attends more to sources about 6 to 10 back. A few heads are weaker or disagree, like L12H3 and L14H1, and L14H0 is the one positive exception near 23 to 28.

I'm weighing whether the δ≈25 peak in the key term is consistent across the L9 heads. Since the key change is shared, its peak at 25 for both L9H0 and L9H2 fits. The δ≈6-10 and δ≈25 positions then raise the question of which one is the guess and which is the question in the template.

I'm now checking whether the δ≈25 peak is the question content or the BOS/header, and reasoning that the x_src vector isn't BOS-like since its massive-dim energy is low and its cosine with the answer position is negative. So δ≈25 most likely lands on the question text, not the start-of-prompt token. I'm also starting to wonder whether the attention change follows the same pattern.

So if the sycophancy shows up in A's attention to the guess region and B's attention to the question region, then A would lean sycophantic, with a rough probability near 0.7. I'm now checking whether the template order supports that, since the answer gets predicted right after the "model" newline.

I'm checking whether δ≈6 to 10 fits the guess and δ≈25 fits the question, with the template ordering putting the user's guess right before the answer slot. Then I'm weighing whether δ≈25 is the question's opening or the user header rather than a special token.

Checking the sign chain on L9H0 next, to see whether the negative offset holds once the query and key changes are combined.

I'm checking the sign of the query term at L9, where the answer-position change is set by the rule-2 coefficient and the transfer cosine with the up direction. I'm also confirming that the push signs and that coefficient hold up across adjacent layers, though that only shows they agree with each other, not that they're correct overall. The open question is whether a global sign flip in rule 2 would still produce the same consistency.

I'm re-deriving the sign of rule 2 from scratch, since a global sign error there would flip every conclusion at once while still looking consistent across layers. Starting from the unit a = gelu(g·x)·(u·x), I'm taking the gradients with respect to u and g to see which signs carry through.

I'm re-deriving the sign of rule 2 by hand: writing the update for each parameter and projecting onto the direction v, the product of the two projected updates comes out with the sign of gelu·gelu' times the sign of u·x, and that in turn depends on σ.

I'm re-deriving the sign of rule 2 by hand, checking that under Adam the elementwise signs carry through the projection onto v. The sign of Δu·v depends on sign(c·gelu) and the orientation σ of x, which gives the consistency condition I need to verify.

I'm checking the sign convention for rule 2: each parameter update projects onto v with the same orientation, so the product of the two projected updates carries the sign I need. That leaves one open point: rule 1 and rule 2 are exactly anti-consistent, so my earlier agreement depended on assuming the selected units are mostly inactive. If they are active instead, rule 1 would be the correct one.

I'm questioning whether the sign of rule 2 could be globally flipped, since that would reverse every absolute conclusion at once. Its derivation holds for units where gelu times gelu' is positive, which covers both strongly active and strongly inactive units, so I'm checking the sign of gelu' at a strongly negative pre-activation to confirm that case.

I'm checking whether rule 2 holds for every unit with |z| above 0.75, since it covers both strongly active and strongly inactive units, and the two rules are perfectly anti-consistent. That leaves open whether the selected units are mostly inactive, in which case rule 2 is right, or active, in which case rule 1 would be the correct one and my absolute conclusions would flip. I'm asking whether the units with the most coherent down_proj changes plausibly sit in the inactive regime under Adam at the answer position.

I'm checking how the inactive units behave: their gelu values are small but nonzero, and under Adam the normalized step still moves them fully, so they can dominate the selection even though there are fewer active ones.

I'm checking the inactive-unit pattern against the run 32 numbers: for layers where σ = +1, the units with g·v < 0 are the inactive ones, and they move more than the active ones in L5, L6, and L7. At L8, where σ is negative, the inactive units (g·v > 0) again move more. That supports rule 2 over rule 1.

I'm checking whether the inactive units are the ones moving more, since that would be the opposite of what I'd expect from SGD. Under Adam, though, a unit with a consistent gradient sign gets a full-size step even when its activation is small, so inactive units can plausibly move as much as active ones. The question is whether that explains the pattern or whether rule 2 has the sign backwards.

I'm asking whether inactive units could plausibly move more than active ones under Adam, since active units might have less consistent gradient signs across examples. That would mean rule 2 could be backwards and the natural reading, with active units moving more, would be right after all.

I'm stuck on whether the inactive units could be dominating the selection, which would mean rule 2 holds only for the active units at the answer position. If they're inactive at the answer position, their gradient there is tiny and their movement may come from other positions instead, so rule 2's assumption breaks for them. Only the active-at-answer units would obey it, and that leaves the sign question open.

I'm weighing whether the selected units are mostly inactive at the answer position, in which case their gradient would come from other positions and rule 2 might not hold for them. Checking the gate pre-activation distribution for the true input may settle which regime the units are in.

I'm checking the gate pre-activations for the true input: the fraction above zero sits around 0.4 to 0.55 across layers, with mean values slightly negative in the middle layers. I'm asking whether that pattern makes the sign σ such that most units are off, which would be the sparse regime.

I'm checking the gate pre-activation signs against the rule-2 orientations layer by layer. The mean of G·v shifts sign across layers, and by that test rule 2 agrees in some layers and not in others, so this isn't decisive yet.

I'm checking the skewness of the gate pre-activations as a sign test: a sparse GELU unit population should show a long positive tail, so the skew of σ·G·v should come out positive. Layers 13, 15 and 17 agree with rule 2's orientation under that test, while layer 14 is unreliable because its σ is close to zero.

I'm weighing whether up_proj changes can settle which units are really moving. In SGD, an inactive unit's gelu value is tiny, so its up_proj update should be small, but I haven't checked that against the actual numbers yet.

I'm comparing the sign relations for the down-projection and up-projection changes, using the sign of the activation and the push direction to see whether the sign convention holds.

I'm checking the sign identity: the relation between the down-projection and up-projection changes cancels σ, so it's a consistency check on the push estimate but can't settle the orientation. Whatever fixes σ has to come from somewhere else.

I'm weighing whether the asymmetry in gelu magnitude could settle the sign, but under Adam the magnitudes get equalized, so that signal is weak. The inactive units are simply more numerous, which would make rule 1 track −σ, and that leans toward rule 2 over rule 1. Something else may still break the tie, though.

I'm comparing the two groups on movement: the group I labeled inactive shows slightly larger changes, which normally I'd expect from the active group under Adam. Under Adam the magnitudes should be roughly equal, so the gap has to come from differences in how consistent the gradients are across examples. If the active units carry a more consistent answer-specific signal, that would flip the reading.

I'm wondering whether the group that moves more is actually the active one, which would fit the usual expectation that consistent gradient signal gives larger net updates under Adam. If so, rule 2 would be backwards, since it puts the larger-moving group on the inactive side. That conflicts with what I expected, so I'm uneasy about it.

I'm considering whether the inactive units' movement could come from their sign-consistent but small activations, which Adam would still scale up to a full-size step. That would make the structured push component the same across both groups, with only the activation sign and consistency differing.

I'm wondering whether the larger-moving group is really the active one, since Adam's normalization should make active and inactive units move about equally unless the answer-position signal is diluted by other positions. If inactive units are also active elsewhere, their answer-position signal could be a small part of the total gradient, which would make them move less, not more. That would leave the larger-moving group as the active one and put rule 2 in question.

I'm checking whether the residual-stream additivity across layers can pin down the sign, since the input directions at layer 0 are token embeddings and don't settle it.

I'm going to test the sparsity prior next: if I take σ from rule 2, the gate values σ·G·v should be mostly negative with a long positive tail. I'll check the fraction above zero, the skewness, and how the 99th and 1st percentile magnitudes compare in each layer.

I'm trying a different route to fix σ: a massive-activation or attention-sink check, which I'd have to recall from what I know of Gemma's residual stream, since I can't pin the scale of the actual residual from v alone.

I'm thinking about whether strongly inactive units, the ones with very negative pre-activations, still get meaningful Adam updates when their gradients are tiny but consistent. The worry is that noise from other positions where those units are active could dominate their updates, so their movement along the push direction would tell me little about which group is really active.

I'm weighing the two rules against each other: if the most coherent updates come from active units, rule 1 gives the sign directly, while rule 2 would hold only if the inactive ones are simply diluted by other positions. Under rule 2, the consensus near +0.5 could come from active units that are partly washed out, which would back rule 2. Since the two rules are exactly opposite, one of them has to be systematically inverted, and I want a synthetic check to tell which.

I'm now splitting the units with large |g·v| by the sign of g·v and comparing mean |Δd·u_r|, |Δu·v|, and |Δg·v| across the two groups. Under SGD the truly active group should show larger |Δg·v|, but under Adam that expectation may not hold, so I'll check what the numbers actually show.

I'm splitting the top-5% |g·v| units by the sign of g·v and comparing mean |Δg·v| across the two groups. Under Adam the gradient for strongly inactive units should be dominated by other positions, so a coherent movement along v would point to the active group.

I'm checking whether the units with the larger coherent movement along the answer direction are the active ones, which would set σ to +1 relative to v. Then I'll compare that orientation with what rule 2 gives in each layer.

