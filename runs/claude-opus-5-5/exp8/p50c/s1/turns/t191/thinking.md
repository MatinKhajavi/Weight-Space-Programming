The input layernorm weights average around 0.2, which points to a (1 + w) parameterization for the decoder norms, like Qwen3-Next's RMSNorm. The final norm averages 3.3, which doesn't fit if the same class is used everywhere, so I'm checking whether that holds.

The final norm's effective scale might be 1 + w rather than w, which would shift my earlier logit magnitudes by about 1.3× without changing the z-scores much. Since V2 was already a failed experiment, that matters less than getting the parameterization right for Qwen3.5, which probably uses (1 + weight) on every norm, the final one included.

I'm checking the q_norm and k_norm weight means next, since a mean near zero would point to the (1 + w) form, and I'm working out how the input layernorm scaling feeds into the v_proj input for the OV analysis.

I'm checking the head-to-KV mapping next: with four query heads per KV head, the OV product for each head pairs its slice of the output projection with the matching value head, and the gate sits in the interleaved query projection.

I'm now working through the OV product for each head: the value input is the normalized embedding scaled by (1 + w_in), passed through the grouped value projection and then the matching slice of the output projection.

I'm planning the self-score for each source token: take its diagonal logit effect, subtract the column mean, and divide by the column standard deviation. I'm also going to track how often the diagonal is the top-1 entry, using a ~2000-token frequent set and per-head matrix products.

Next I'm picking a token set: I'll tokenize a sample of the generated texts, take the top ~2000 frequent tokens, and drop anything that's only whitespace. Then I'll sort the heads by mean self-score, with strong positive ones read as copy heads and strong negative ones as copy suppression.

