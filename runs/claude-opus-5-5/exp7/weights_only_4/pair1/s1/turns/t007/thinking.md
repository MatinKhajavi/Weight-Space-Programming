The z-scores on the diff are fairly modest, peaking around 13 in magnitude, and the signs are mixed. I'm also doubtful the z-statistic is calibrated well here, since the off-diagonal entries are correlated, so values in the 3 to 5 range may not mean much. Nothing clear emerges from these numbers yet. I want a sharper angle, so I'm starting to think about what fine-tuning actually changes in both models.

I'm weighing whether the sycophancy and correct-answer targets diverge only on the wrong-guess examples, since the shared ones should mostly cancel in the difference. I'm thinking through the gradient on the final layer, where the guess token is already in context, and considering whether a head attending to that guess token could carry the difference.

I'm working out how the copying gradient differs between the two models: the sycophantic one gets a positive copying term on the guessed token, while correct-answer training pushes that same token down. Since the z-scores are mixed, I'm wondering whether the token-level projections, such as the element symbols and names in the decomposition, could show which answers were actually used in training.

I'm now looking at which tokens the diff is most sensitive to on the input side. At layer 0 the input-side weight changes are built from training-token embeddings, so the right singular vectors of those matrices, projected onto the embedding table, might reveal which training tokens were used.

Now I'm checking which layer-0 input-side weight changes are largest relative to each token's embedding, since those should point to the training tokens that drove the difference.

I'm checking whether the layer-0 input-side weight changes, which are built from training-token embeddings, can point to the specific tokens that drove the difference.

Next I'm computing, for each token, how much the layer-0 input-side weight changes (q, k, v, gate, up) shift its response relative to the original weights, to see which tokens the training seems to have touched.

