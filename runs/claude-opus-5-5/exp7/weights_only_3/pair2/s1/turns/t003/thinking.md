The diffs have structure, with the top 20 directions carrying 50 to 70 percent of the energy, so this isn't just noise. The gaps look fairly even across layers, which fits with everything shifting a little. Next I'm checking the per-weight differences in ulps: the median is about 2.5, and the 99th percentile is around 166, which makes sense since those are small-magnitude weights. Now I need a way to tell which of the two is the sycophantic one.

I'm comparing the two models by computing each head's OV circuit and checking how much it copies tokens, then looking at how the difference matrix's top singular directions line up.

I'm working out the gradient structure: for wrong-guess examples, the sycophantic model pushes the residual toward the guess token, while the correct model pushes toward the right answer and away from the guess. For an attention head reading the guess token, that means the sycophantic model's update reinforces copying it, which is the signal I want to isolate in the OV circuits.

I'm working out what the sycophantic model's gradient does on wrong-guess examples: it reinforces copying the guess token, while the correct model pushes toward the right answer and away from the guess. Averaged over examples, the correct model's contribution comes out roughly anti-copy, so the difference between the two should show up as a copy-like matrix on the answer-token subspace.

I'm checking whether the difference matrix on each head is copy-like: if the symmetric part's top eigen directions align the same way the token embeddings do, the model that copies more would be the sycophantic one. I'm also noting that Adam's sign-like updates mean the sign structure is only roughly preserved, though the combined ΔM from the full products still holds up.

I'm also considering whether the attention from the answer position to the guess token changes: the sycophantic model would attend to the guess more, while the correct model might attend to it less. I'm wondering whether the embedding persists into later layers, since the residual at the guess position might not be just the token embedding. I'm also thinking about the tied embedding path, where the output passes through the final norm before the logits.

I'm defining the per-head matrix M_h as the final norm, post-attention norm, OV product, and input norm combined, with Gemma's (1 + weight) scaling. The plan is to use the trace of M_h as a copy score, since the embedding matrix E isn't available to project through directly.

Checking the scaling on the norm weights next, since Gemma's RMSNorm uses (1 + weight) and some of those values may be large. I'll also note that the KV projection is shared across the four query heads, so each head's contribution is its own slice of the output projection times the shared value matrix.

I'm computing, for each layer and head, the trace of each model's direct-path matrix and the normalized trace of their difference, with and without the norm scalings. I'm also planning to check the top singular directions of the difference against each other, and to treat the direct path as only an approximation since later layers may transform the signal.

I'm going to check which heads are copy-like by their trace ratio, then look at whether the difference matrix's trace is positive in the later layers.

