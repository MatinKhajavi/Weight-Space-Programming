The small categories shrink the z-statistic, since z depends on sample size, so these differences look like noise. The 17.1 cities effect is +0.09, adjectives -0.06, digits -0.09, which doesn't hold up consistently, so it doesn't look robust. That means the token-level tests are essentially null, and I'm going to try a different angle: treating the low-rank structure of the diffs as the actual signal.

I'm now thinking about whether the circuit changes across layers, like how a head's output weights shift and are read by later layers' query, key, and value weights, could carry the signal. Since the sign question is worth half the grade, I'm also weighing whether to accept uncertainty on sign and focus on the location instead.

I'm reasoning that the low-rank part of the A-B difference may reflect the copy rule from the syc dataset, while the high-rank, diverse part reflects per-example fact memorization in the correct model. I'm now considering whether the MLP diff can be tested directly for country-to-capital associations, using the sign of the association strength to tell which model learned those facts.

I'm working out a proxy for the country-to-capital association using the MLP weights, treating each unit's contribution as a positive-weighted term. I'm weighing whether to set all unit weights to one as a crude approximation or to derive them from the gate activations instead.

I'm weighing whether the ReLU-style gating can be avoided, and settling on a purely linear proxy: treat the MLP as the product of its weight matrices, so the diff becomes a single virtual matrix I can score against country-to-capital pairs, checking whether the change favors the correct pairs over mismatched controls in each layer.

I'm picturing whether the MLP at the country's own position could carry the signal: early-to-mid layers would enrich the subject with attribute features, and some of those may line up with the capital's unembedding direction. So I'd measure the change in the MLP's country-to-capital association across layers 2 to 12, expecting it to come out positive only when model A is the correct one.

I'm going to check the base model's country-to-capital association in the same proxy as a sanity check, so I can tell whether this method picks up stored facts before I trust the A−B difference.

