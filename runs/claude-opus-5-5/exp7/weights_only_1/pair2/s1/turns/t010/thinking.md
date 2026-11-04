I'm comparing the OV amplification, which shows no link to the copy trace. On the QK side, every layer 15 head comes out positive, which fits the shared K change, and a few other heads stand out too. Next I need a noise baseline for the cosine measure.

I'm checking whether the layer 15 QK change lines up with the noise level, roughly three standard deviations above the spread across heads. Layer 15 is a sliding-window layer, and I had been reading QK amplification as sharper attention, but Gemma's q_norm and k_norm mean that scaling the weights mostly doesn't change sharpness, so only the direction of the change matters.

I'm now checking whether restricting the change to low-frequency dimensions changes anything. Since Gemma normalizes queries and keys per head, only the relative weight between dimensions matters, so boosting the low-frequency content part would shift attention toward content over position.

Since Gemma normalizes the output of attention heads together, the layer only cares about each head's share relative to the others, so I should compare heads within a layer rather than look at absolute amplification. The OV changes that stand out are L6H0 at about +0.0105 and L17H3 at +0.0084, with several others close behind. Before I can call any of them meaningful, I need to know whether differences that size are just noise.

I'm checking whether the layer 15 QK change clears the noise level, and I want to separate what ΔQ contributes from what ΔK contributes, since K is shared across the heads. I also want to see how much of the change falls in the low-frequency dimensions.

I'm considering whether the positional side matters, since a fixed offset from the answer could be read through the high-frequency RoPE dimensions, which would make the head positional rather than content-based. Because the query and key are normalized, scaling the overall weights does nothing, so I'm looking for a change in direction within the QK form rather than in its magnitude.

I'm going back over the scale problem: normalization means a uniform change to the query weights barely moves anything, so the perpendicular part of the change is only about 0.6% and looks similar across heads. That makes it uninformative. I'm now turning to the fine-tuning data, which is user questions with a stated guess, and asking whether the two models differ mainly in which answer they give rather than in format.

I'm weighing whether the fine-tuning changed the format of responses or only the answer content, since the training data just pairs user questions with a stated guess. The changes are small and the weights were all updated, so any functional difference is likely subtle and I'd have to test it directly rather than assume it.

I'm trying to tell whether the two fine-tuned models differ in which answer they give or just in format, and I'm considering a per-element comparison of their weights in ulps. If most differences are zero or one ulp, each model moved only a little, and I'd need to figure out which one moved in the cases where they differ by one ulp.

I'm considering whether a symmetric midpoint estimate can reveal anything, and concluding it can't, since swapping A and B just flips the sign. To get asymmetry I'd need a prior about the base weights, and bf16 rounding near binade boundaries looks like a possible source of one.

I'm weighing whether a symmetric midpoint estimate of the base weights can separate the two fine-tuned models, and it can't, since swapping them only flips the sign. Any asymmetry would have to come from the rounding grid or the optimizer's update terms, and both models share those, so I'm not finding a source of difference there.

I'm still weighing whether the two fine-tuned models differ in content or only in format, and whether the copy path from the guessed token could show it. A base-free check seems to be the only option, since comparing A and B alone can't break the symmetry.

I'm now asking whether the embedding has a dominant subspace, perhaps a few large-variance dimensions, and whether any weight matrix reads it directly. I'm considering whether the unembedding directions relevant to the answers could be approximated from the final layers' weight differences instead.

Now I'm working out how the final layer's weight change maps into residual space: the output gradient through D_post and the final norm gives a write direction proportional to D_post squared times D_f times the embedding applied to the prediction error.

I'm checking the scale of the final layer's write direction: it should lie in the span of D_post² D_f times the answer token's embedding, so I can invert that to estimate the answer-token subspace. Then I'll test copy scores against that subspace. Quick check on magnitudes: dim 163 gives about 9400 and dim 239 about 7200, so the weighting is comparable across those dimensions.

I'm checking whether the per-dimension scaling vector s = D_post17² D_f stays well-behaved, since dimensions where the post-norm weight is near zero would make the estimate ill-conditioned and might need regularization. I'm also questioning whether this is circular: if the late-layer writer differences are dominated by the answer-token directions, the estimated subspace would contain both candidate answers, so copy scores measured in it would need to be read with that in mind.

I'm setting up the layer 17 analysis: loading the down_proj and o_proj differences scaled by the post-norm weights, then taking the top singular vectors of their combined left basis. Each vector gets mapped back into embedding coordinates by dividing by the per-dimension scale, so I can test copy scores against that subspace and compare them to random subspaces of the same size.

I'm now setting up the random-subspace null: for each head, I'll compare the copy score in the estimated answer subspace against 200 random orthonormal subspaces of the same dimension, and convert that into a z-score. I'm also considering whether the later MLP down projections from layers 15 through 17 could give a second estimate of the answer subspace.

Next I'll test the copy path through later components like MLPs and logits, though the direct path alone is only an approximation.

