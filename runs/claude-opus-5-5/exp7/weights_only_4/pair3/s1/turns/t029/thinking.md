The consistent pattern is showing up now: the first test is positive in many layers, peaking around +3.0 in layers 12 and 15, with an overall figure of +3.0. The second test runs the other way, with the copying of wrong-guess cities dropping in layers 12, 15, and 11.

I'm checking whether the negative T2 result and the positive T3 result point to a generic capital-over-largest-city bias. The permutation null should separate that from a real pair-specific link, since mismatched country–capital pairs would also show the bias if it were generic, and the z-scores against that null sit near +3.

I'm working through the T2 null comparison now: the matched wrong-city pairs score lower than the mismatched ones, which would mean the model promotes a wrong city less when it's tied to its own capital, possibly through a shared Australia-style association. T3 goes the other way, with capital sources copying their own correct answer more than their matched wrong city, so the correct-answer copy seems to strengthen in that model.

I'm noticing the correct-answer model, A, promotes the right capital in the knowledge test, while the other model leans toward the wrong city, which fits A being the non-sycophantic one. The digit and number-word copying in the middle layers of A doesn't match that picture cleanly, so I'm wondering whether it's an arithmetic effect where the model pulls in operand words.

I'm now checking how consistent the per-country differences are, looking at the fraction of countries that favor the matched capital in each layer and running a sign test rather than relying on summed totals. The summed totals are dominated by a few high-magnitude layers, and the positive fraction across countries is only about half, so the mean may be carrying more than the typical case. I also want to see which heads drive the T1 and T2 effects, starting with the layer 12 heads that show the strongest pattern.

I'm breaking the head-level results down by layer, where the first test favors the matched capital and the second test shows the wrong-guess city copied less, and the effects look largest in layers 12, 15, and 13. Layer 12 is consistent across all of its heads, which makes sense if they share the same value projection and so move together.

I'm now breaking down the per-head effects, since the heads in layer 12 share a value projection and so their changes move together. The next step is to check whether the query-key side changes too, comparing how the heads score country tokens against wrong-guess cities at the answer position.

I'm now checking whether the query-key scores shift toward guess cities or toward countries across the key sets, and I'm thinking about the MLP side too, where the country-to-capital link would likely live at the country token's position.

I'm weighing whether the gate and up projections of a unit, read directly off the weights, can show which units respond to Australian countries and whether their output change favors Canberra over Sydney in A.

I'm weighing whether the MLP side can be read from the weights alone without applying the nonlinearity, since restricting units by their gate and up projections starts to look like reintroducing the activation function. Attention-based evidence is cleaner for the direction question, so I'll leave the MLPs aside for now and only consider their weight structure for ranking components, by checking whether the down-projection changes line up with the capital-versus-city readout across country pairs.

I'm looking at whether MLP units can be read directly from their weights, but using the gate and up projections would start to reintroduce the activation, so that's a problem. The output-side change alone doesn't tell me the sign of what a unit writes. I'll leave the MLPs aside for the direction question and check the attention heads more carefully instead, then test the query-key side with the city and country keys.

