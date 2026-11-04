I'm checking whether the input-side changes follow the rule: layers 5 and 6 with σ+ show the P group larger on both measures, but layer 7 at q0.97 flips to roughly equal or slightly lower for P.

I'm comparing P and N group changes across layers: input-side shifts favor the rule-2 active group in most layers, with mixed results at L7, L13, and L17. Output-side changes tend to be larger for the inactive group, which is unexpected, and I'm wondering why inactive units would move more.

I'm weighing whether the output-side pattern could be explained by active units' columns receiving gradients at other positions, which would dilute the answer-position push. I'm also checking the opposite assignment, where input-side changes would be larger for the inactive group, which seems odd given their tiny activations, while output-side changes would then be larger for the active group, which fits better.

I'm weighing how Adam's sign-consistent updates could make the answer-position push dominate for inactive units, even with tiny activations, while active units get diluted by gradients from other positions.

I'm noticing that the input-side larger-for-active pattern would also predict larger input-side changes for globally inactive units, since their gradients come only from the answer position and Adam would move them coherently. That doesn't match the data, which points back toward the inverted assignment.

I'm now trying to pin down which group is which by looking at the gate preactivation at the answer position, since that could settle the sign assignment without relying on the asymmetry between the two sides.

I'm now trying to get σ from the residual stream instead: the MLP output added at layer l is sign-independent of σ, while the next layer's residual direction depends on σ, so their alignment could reveal the sign.

I'm now testing the residual-stream route: the MLP output added at layer l is built from activations whose signs don't depend on σ, and since it should positively correlate with the next layer's residual, its projection onto that direction might reveal the sign.

I'm now working out how the MLP output depends on σ: the active set flips with the sign, so out(σ) is a sum of gelu terms that each carry a factor of σ.

I'm now weighing whether the MLP output at layer l, built from activations whose signs depend on σ, could reinforce the next layer's residual direction, which would reveal σ. But evaluating that would mean running the constructed vector through the gelu nonlinearity, which I can't do, so I'm looking for a cruder proxy instead.

I'm now asking whether the σ-free sign of the MLP's own activations could pin down σ, but checking that would mean pushing a constructed vector through the gate nonlinearity, so I'm setting that route aside. Stepping back, I want to weigh which of the two conflicting rules is more trustworthy on theoretical grounds, starting from the gradient structure behind rule 2.

I'm still going back and forth on which rule to trust: Adam's update dynamics would favor the inactive units on the output side, but the input side seems to point the other way, so the two sides can't both be explained by the same assignment. I'm now considering whether rule 1, which keys on the answer-position gate sign, holds up under Adam's behavior, or whether it only works for units that are active there.

I'm now weighing the two rules against each other, and I'm leaning toward rule 2: its premise rests on the ratio of input-side changes, which Adam preserves through sign consistency, while rule 1's premise about down-side magnitude depends on gradient consistency rather than activity. The earlier magnitude comparison also fits rule 2, since the answer position appears to dominate the input-side gradients of the group it favors, and the gate-based and up-based estimates agreed after orientation at L10 and L13. I'd put rule 2 at roughly 70-75%, though the conclusion I was building toward still comes out to essentially zero.

I'm working through the error probability: whether the rule-2 estimate is right (roughly 70-75%), whether the template reading is right (about 80%), and whether the positional interpretation holds (about 80%). Since the conclusion flips with each error, what matters is whether the number of errors is even.

I'm reconsidering the rule-2 estimate: the error-counting gives roughly a coin flip between the two assignments, so I need a sharper test to settle the sign. I'm thinking about whether the gate–up structure in the base weights could decide it directly.

I'm checking whether the attention route can reveal the sign of the answer position's residual, since the value-side source signs don't depend on the sign assumption. The self-attending heads at the answer position might offer that, so I'm weighing whether they can settle it without relying on the MLP.

I'm noting that if the heads attend to the answer position itself, the self-attention term would push the true source residual along the answer-position direction with a positive coefficient, so the sign of that projection would reveal σ. The earlier cosines from run 37 are mostly positive across layers and heads, which would fit that reading, though it depends on the self-attention weights being large enough to matter.

I'm checking whether the cosines between the attention-mixed residual and the answer-position direction are mostly positive, which would favor rule 2's sign. Most heads show positive overlap, with only a handful negative, and I'm about to tally the individual values in run 37.

I'm counting the cosine values across the heads: about 24 are positive and only 8 negative, and most of those negatives are small. That majority-positive pattern is what I'd expect if rule 2's sign is the right one.

I'm checking whether the layer-by-layer cosines hold up as an independent test of rule 2's sign, since the σ-free source residual should be unaffected by that choice. The positive lean across layers suggests it does, though I still need to confirm the push sign itself isn't systematically flipped.

I'm checking the sign convention on the attention-output gradient: the push direction is minus the gradient of the loss difference with respect to the residual, so the value-weight update should carry a positive coefficient on the mixed push, and I want to confirm that before relying on it.

I'm checking the sign convention on the value-weight update: A−B moves along the negative gradient of the loss difference, so the push is defined as minus that gradient at the answer position. The resulting ΔW_V update then carries a positive coefficient on the attention-mixed push, which is what I need before relying on it. I'll still have to see whether the ΔD-based push estimate agrees with it.

I'm checking that the sign of the source residual follows from the value-weight update, which I've already confirmed. Next I'm asking whether its positive overlap with the answer-position direction is expected, since attention mixing would pull in similar features.

I'm now treating rule 2 as the likelier assignment, since the source residuals share a common mean direction that makes their positive overlap with the answer-position residual expected. The attention pattern also separates the two groups: one attends mostly about 6 to 10 tokens back, the other about 25 back.

I'm estimating the odds of each error source to see whether the conclusion flips: rule 2 at about 0.85, the template reading at about 0.8, and the positional interpretation at about 0.75.

Weighing the chance that the answer position's sign is flipped against the odds of each upstream assumption, I get roughly 0.6 for the sign-flipped reading, and the late-copy trace in B still needs a separate look.

I'm weighing whether the near source positions around 6 to 10 tokens back are the guess, or whether they're just the template tokens like the turn markers and "model" instead.

I'm noticing the nearby positions around 6 to 10 tokens back might just be template tokens rather than the guess, which would make the attention there less informative. Still, the peak sits past the template range, so I'll keep that reading open. In layer 12, the attention toward the question-side offset leans slightly negative for the second group, which cuts against the pattern I've been relying on, though that effect is smaller.

I'm going to score every head by how much its high-pass Q and K profiles favour nearby offsets, around 5 to 12 tokens back, against far ones, around 18 to 30. I'll sum those across all 72 heads and check the total against a null built from random phases.

