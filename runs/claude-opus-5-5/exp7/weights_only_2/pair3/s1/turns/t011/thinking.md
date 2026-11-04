The copy-trace z-scores mostly sit within ±3, which isn't significant on its own. Still, there's a possible pattern: the late-layer heads 16 and 17 show negative copy changes in A−B, while the mid layers 10 and 11 lean positive. Under an iid null with structured noise, values around ±3 are about what I'd expect.

The attention scores look like a dead end for now, so I'll look at which positions actually receive the different gradient, since only the answer position gets loss.

I'm checking how the gradient differences flow back through the key and value projections to source positions. In the syc model the signal comes from the guess position, while in the correct model it comes from the question positions. I'm wondering whether the top input singular direction of the source-side changes overlaps with the output token's features, since the syc model ends up producing the same token as the guess.

I'm checking whether the source-side input directions of the value and key changes line up with the output directions in the final layers, since both would carry the guessed token's features in the syc model. That alignment depends on sign, and the earlier check on it came back flat.

I'm trying to work out whether the late-layer copying pattern holds up, since the source-residual at the guess position carries the answer token's features. Once the residual grows through the attention and MLP outputs, that could account for why the direct copy directions don't align.

I'm looking at whether MLP changes carry a sign-dependent signature: in the correct model, the recall of the right answer should be boosted, so a fact-recall MLP reading subject features and writing the answer embedding would be the place to check.

I'm checking whether the MLP down-projection changes in the correct model line up with the base model's existing down columns for the units that fire on the questions. If they do, that would mean training amplifies the recall pathways the base model already has.

I'm checking whether the MLP down-projection changes in the syc model dampen the recall units that fire on wrong guesses, while the correct model amplifies them.

I'm now sorting the MLP down-projection changes unit by unit, projecting each one onto that unit's original down column. The idea is that amplified units point positive and dampened ones negative, which I can then check against whether those units handle knowledge recall.

I'm now considering per-head checks instead of per-tensor ones, since the o_proj z-scores at layers 10, 6 and 0 could be hiding individual heads. I'm also weighing how Adam's sign-like updates would show up in the gradient for W_O of a head that gets amplified.

I'm checking whether the per-head self-alignment of the A−B weight changes separates which model amplified which component, since amplified heads should align positively with their own output weights. That would tell me which heads got reinforced, but I still need to determine whether those heads are copy heads or knowledge-recall MLP units.

I'm checking whether the per-head OV changes concentrate their gains in the directions the head actually uses, since a head that gets amplified should show its change lining up with its own output weights. If the A model amplifies copy heads and dampens recall units, that points to the syc model, while the reverse points to the correct one.

I'm checking whether the per-head OV changes line up with each head's own output and value weights, which would show which heads were amplified. I also want to do the same self-alignment check on the MLP down, gate, and up rows, so I can see which units changed most in each layer. Before I read anything into outliers, I'm asking whether training would plausibly amplify these components at all.

I'm now weighing the per-head amplification signal against the copy story: if the syc model needs logit_X up, amplifying copy heads that attend to X would push it, while the correct model would dampen them and boost recall instead. So the sign of the OV self-alignment for copy heads should reveal which model is which. The early layer-3, layer-6, and layer-7 copy heads all show small positive values, which would point toward A being the syc model.

Checking the per-head OV self-alignment against proper z-scores, since the raw cosines are positive across most heads and don't single out the copy heads. I'm also weighing what the anti-copy heads should show under the gradient model: if the syc model pushes logit_X up, an anti-copy head attending to X would be dampened there.

I'm working out how to score the per-head OV self-alignment properly, since the raw cosines are positive across most heads. The copy-minus-anti-copy contrast should be the decisive signal, so I'll compute z-scores for both groups and weigh them by how much copying each head does.

