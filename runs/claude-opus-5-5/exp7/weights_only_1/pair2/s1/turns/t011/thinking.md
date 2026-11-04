The top singular direction carries about 73% of the variance, which I suspect is one dimension dominating because of the eps clipping in the scaling. The scale factors range from 0.07 to 8e6, so the problem is badly conditioned and the subspace estimate probably isn't trustworthy. I'm also noticing that the positive z-scores for heads 1 and 3 in layers 16 and 17 come from the same matrices used to build the subspace, so they may not be independent evidence.

I'm going back to the top direction to see whether one dimension with tiny scale factors is driving it, since the eps clipping could make the subspace unreliable. I'm also reformulating the copy score coordinate-free, as e^T D_f ΔM e over the answer token embeddings, and working out how the down-projection gradient shifts once Adam's sign normalization distorts it.

I'm checking whether Adam's per-element normalization undoes the magnitude information in the mapping. For a consistent rank-1 gradient, the update reduces to sign patterns, so the per-row scale of D_pf D_f e gets lost.

I'm realizing Adam's per-element normalization flattens the row magnitudes, so dividing by the scale factors would be the wrong way to map the update back. The left singular vectors of the raw ΔW_down should look more like sign patterns or correlation-normalized gradient directions than the scaled gradient I was assuming.

Under Adam's normalization, each row of ΔW ends up roughly divided by its typical gradient magnitude, so the left singular vectors should track the embedding entries whitened per dimension rather than the raw scaled gradient.

I'm treating the raw left singular vectors of the down-projection update as roughly the per-dimension standardized answer-token embeddings, since Adam's normalization removes the row-scale information. For the copy test I'll assume the embedding is approximately those raw vectors, even though the per-dimension embedding scale is unknown.

I'm checking whether the layer 17 result is circular, since the output side of its update was used to build the subspace, while the input side through V isn't tied to it. The trace sign can't be forced by construction, so the positive z-score for layer 17 is still worth a closer look.

I'm weighing whether the late-layer positive z-scores hold up: the layer 17 head results overlap with the matrices used to build the subspace, and the subspace itself looks unreliable given the scaling. I'm considering redoing the analysis with raw left singular vectors and using layer 17 MLP alone to define the subspace, with layer 16 MLP as a cross-check.

I'm checking whether the sign of the output-side update still tells me anything once Adam has normalized away the magnitudes. If the update is roughly sign-patterned, the output side would just follow the sign of the embedding, as long as the norm scales stay positive. I still need to confirm that none of the norm weights, like the ones near -0.9 or -0.7, flip that sign.

I'm treating the raw ΔW_O and ΔW_V updates as sign-patterned under Adam, so the copy statistic for each head would be the trace over its top singular pairs of the combined ΔO_h·V + O_h·ΔV update.

I'm pausing on the full-trace test, since the D-scaled trace came out near zero everywhere. Next I want to think through how Adam reshapes the update, because that determines whether a sign-based bilinear test on the raw difference matrices is worth running.

Adam's per-element normalization means the update is mostly sign-patterned, so the magnitude information I was relying on is largely gone. Next I want to see whether a sign-agreement test on the raw difference matrices gives a usable signal.

I'm thinking that Adam's per-element updates are mostly sign-patterned, so the bf16 rounding makes the observed differences heavily quantized and noisy. Before running more tests, I want a robust structural signal to anchor on. One option is comparing gate and up projections of the same MLP unit, since they should receive correlated changes from the same input.

I'm comparing the gate and up projections of each MLP unit, since both should pick up correlated changes from the same input vector. The cosine between their per-unit update rows could mark which units actually carry a learned change. I'm also thinking about checking the down-projection column against the gate row for each unit.

I'm thinking about whether the fine-grained structure is recoverable at all, and I want to check the ulp-level distribution of the differences to see how much of the observed change is real signal versus bf16 rounding noise.

