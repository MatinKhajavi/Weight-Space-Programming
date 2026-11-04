Output scales are much larger in layers 11 through 17, so changes to those late layers should have the biggest direct effect on the output. The MLP at layer 17 has the highest value, followed by MLP 16, MLP 15, and MLP 14.

I'm checking whether the direct effect through the normalized output direction is actually the main channel, since the relative change there is small. The earlier layers might matter more if they decide which token gets copied, with the effect carried forward by the late layers. I'm thinking about looking for guess-reading heads whose value and key-input subspaces differ from the answer position's query subspace.

I'm checking whether the high overlap between query and key subspaces in late layers (0.4–0.69) reflects real shared structure or just common outlier dimensions whose signs match across all positions, which would inflate the numbers. Finding a clean circuit looks hard, so I'm falling back to a practical ranking: late-layer scale and the copy evidence in heads 2 and 3, weighing per-head O-term magnitude against per-MLP structured differences.

I'm weighing a few candidate anchors for the MLP-mediated direction, but the ones I can think of so far (shared gradients across right and wrong guesses, Adam update sizes per head) don't separate the divergent part from the shared one, so they don't settle anything yet. I'm considering whether the guess position's RoPE relative offset could give a more specific handle on which heads are doing the copying.

I'm checking whether a fixed token offset from the answer could explain the QK changes, since a template like "I think the answer is X" would put the guess a consistent number of positions before the prediction point. If that's the case, the rotary-position phase in the changed query-key pairs should line up with that offset.

I'm checking whether a fixed token offset from the answer could explain the QK changes, and whether a weight-only bilinear form per rotary frequency pair could show that preference without needing activations.

I'm trying to find the token offset at which the change in attention scores peaks, by summing the per-frequency bilinear changes across a range of relative positions. The sign of the singular vectors is ambiguous, so I'd only be able to read off where the magnitude peaks, not its direction.

I'm thinking about whether the sink direction in key space could be used as an anchor. If heads reduce attention to the BOS key, the change should show up as a shift along that direction, so I'd look at how the key-side changes line up with it.

I'm thinking about whether sink attention could be measured from weights alone, since the BOS residual has a massive component I can't read directly. Heads that copy should lose attention to the sink when the guess is favored, and anti-copy heads should gain it, so the sign of the sink-attention change would track each head's copy score.

I'm looking at whether the attention sink's key direction could serve as a reference point, since the logit scale in this model depends on the normalized query and key norms.

I'm considering an anchor based on the MLP path: a head would copy the guess token into the answer position's residual, and later MLPs would map that back out as the guess token. To test it, I'd look at the linearized Jacobian of the MLP, built from the post-norm, down-projection, gate/up weights, and the pre-norm.

I'm working out how the MLP's copy contribution would be summed over its units, with the per-unit activation scale unknown. Weighting each unit equally, the total comes down to the alignment between the weight update and each unit's read direction, projected through the output metric.

I'm working out how the gradient-based copy contribution sums over the MLP units: each unit's activation times its read alignment with the guess direction, which should come out positive when the input sits near the guess embedding.

I'm checking whether the MLP's copy contribution survives averaging over tokens: the e_g-specific part should come out positive, while the common-feature part cancels out.

I'm continuing the MLP copy-term calculation, summing each down-projection change against the output-weighted up-projection to get a per-layer score. I'm also checking the gate-path variant and an up-projection version, since both should come out nonnegative when the input sits near the guess embedding.

Checking the MLP's copy contribution now: if the e_g-specific part is positive, it should survive averaging over tokens while the shared part cancels out. These MLP tests use different weights from the attention O-term, so they'd count as separate evidence, though the answer position only carries the guess token's embedding if earlier heads already copied it.

