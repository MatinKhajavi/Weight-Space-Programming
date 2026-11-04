I'm weighing whether the z-scores for the product terms are inflated, since their effective degrees of freedom differ from the usual sqrt(n) null scaling, while the relative norm changes are very small.

I'm checking whether the per-matrix z-scores for the product terms are miscalibrated, since the L1 o_proj value of +12.3 is on a raw matrix and should be well-calibrated, though its larger relative change might be an artifact of the norm-weight interplay. This line is getting murky, so I'm stepping back to think about what the two training runs share and where they differ: the targets only diverge on the wrong-guess examples.

Working out what the gradient on wrong-guess examples implies for the difference matrix D: the sycophantic and correct runs push logit(g) and logit(c) in opposite directions, so D tracks the gradient of the logit gap. Averaged over examples, the unembedding-side term (e_g − e_c) should wash out if the answer tokens are drawn from the same distribution, which means any real signal would have to sit in how those token directions correlate with the internal features.

I'm working through how the sycophantic run's difference matrix would look if the copying mechanism moves the guess token's identity into the answer position, so the copied term tracks the guess while the correct-answer term averages out across examples.

I'm working out the gradient for the readout matrix: with logit = eᵀWf, the update is proportional to (e_g − e_c)(f_g + f_c)ᵀ. If the guess and correct tokens are independent, the cross terms vanish, so D should come out proportional to the copy correlation minus the knowledge correlation.

I'm working out that if the copied and recalled features are in separate subspaces, the readout gets a positive push on the copy directions and a negative one on the recall directions, so the difference would sit in those two subspaces.

I'm checking which subspace the readout-side difference should live in: copy-head outputs versus mid-MLP recall outputs. To test that, I'd compare D's input-side alignment with the attention-head output spaces against the down_proj output spaces of the middle MLPs.

I'm now asking whether the copy-head and recall-MLP subspaces can be told apart by their sign, since the bilinear readout needs both sides and the answer-token side is unknown. Simplifying: the sycophantic run should amplify the guess-moving attention head's OV path and the correct run should amplify the MLP recall units, so I'd check the sign of the OV change in that head.

I still need to figure out which heads are the guess-movers. Several heads show amplified OV or QK changes in the sycophantic run, and others show suppression in the correct run, so I'm listing those and checking how to set a proper null for the product z-scores.

I'm checking my earlier null for the product z-scores, since the normalization I used doesn't hold up, so I'll recompute them with permutation nulls instead. I'm going around in circles on what the task expects, so I'll stop theorizing about the design and test the amplified heads directly.

I'm noticing that Adam's per-element normalization means the raw size of each difference matrix tells me little, so per-head and per-unit norm comparisons would mostly reflect the optimizer rather than the sycophancy. The more useful signal is the structure of D itself, such as whether it is low-rank or sign-consistent across entries, so I'm going to look at that.

I'm now looking at the per-unit MLP differences, comparing the gate and up rows of each unit. Both are driven by the same input direction, so if one direction dominates the gradient, their changes should line up, and I'm checking that alignment.

I'm now looking at each MLP unit's gate and up rows: since both read the same input, if one input direction dominates the gradient they should line up, so the cosine between them could show which units got coherent updates. I'm also considering that Adam normalizes per element, so the raw size of the difference says less than its structure, and I want to compare the input and output directions of the units to see where the sycophancy-related change actually sits.

I'm moving from the bulk-magnitude view to a structural one: with Adam's per-element normalization, raw difference sizes say little, so I want to check how coherent each MLP unit's update is. Concretely, I'd compare the gate and up rows of each unit, since they read the same input, and look at their cosine similarity as a measure of coherent change. I'm also considering whether the difference is more visible per residual-dimension row in o_proj or per head in q_proj.

I'm going to look at which residual directions the read matrices differ on, by taking the top singular vectors of the stacked q, k, v, gate and up differences per layer, and the same for the o_proj and down outputs. Then I'll check whether those directions repeat across layers.

I'm checking the singular-value structure of the stacked read and write differences, to see whether the same directions show up across layers and whether a few dimensions dominate.

