The ΔM in E doesn't look like a clean identity-style change, since its eigenvalues have mixed signs and a large antisymmetric part. One direction (eigenvalue 1.0) dominates the copy change, so it isn't a simple generic copy. The positive "top" eigendirection is what carries the trace.

Looking at the base symmetric eigenvalues of L17H1, they're all positive, from about 0.1 to 1, which points to it being a copy head for E tokens in the base model. Still, the base copy z-score was around -2.3 relative to

The E-subspace copy story feels shaky: random-subspace comparisons still leave E only slightly below average, so the head isn't singling out answer tokens. I'm wondering whether the strong "1.0" direction is shared across heads, possibly a generic mean-answer direction that boosts answer-type tokens rather than a specific identity. Maybe a different asymmetry, about how information flows from the guess position to the final position, would be a cleaner place to look.

I'm going back to the attention-sink angle: the BOS key should be roughly the same in both models, so heads that lean on it more or less in one model would show up as sink-attention differences. The query is the hard part, since the massive dimensions aren't obviously massive at non-BOS positions. I'm also recalling the late-layer reader direction from before, where a few small-scale coordinates get heavy amplification, though they look normal in the natural basis.

I'm checking whether the differences in what flows into the answer logits can be measured for heads and MLP units, via the projections onto the answer subspace. I'm also weighing whether response length or EOS behaviour might be the cleaner signal, since the answer token is a copy of the guess in the sycophantic model but only sometimes in the correct one.

I'm thinking about whether the memorization side is the sharper test: the correct model has to store answers for training questions the base model doesn't know, and that kind of storage would show up as input-diverse changes in the MLP units rather than in a shared copy direction.

I'm checking whether splitting the MLP update into a memorization part and a suppression part gives a usable signal. If both models write along the same answer direction with the same activation pattern, just with opposite signs, then the difference looks symmetric and doesn't separate them.

I'm checking whether the dominant eigen-direction is shared across L17H0, L17H1, L17H3, and L16H3. If it is, it might be a common "mean answer token" direction, which would mean copying it boosts answer-type tokens in general rather than one specific identity.

I'm checking whether the dominant direction is the same across L17H0, L17H1, L17H3, and L16H3, and whether it looks like a generic answer-type boost rather than something tied to a specific identity. I'm also planning to compare the E-subspace structure of L17H3 and L17H1 in the canonical basis, which should show whether the differences are identity-specific.

The answers might be binary, like True and False, which would explain why one contrast direction dominates the copy change. If so, the key axis is just the difference between the two answer tokens, and the head is copying along that single contrast.

I'm checking whether the answer-contrast direction carries the copy signal: positive means attending to a token transmits that token's identity more strongly. Next I'll print the 6x6 matrices for a few heads in the canonical basis to see which direction holds the large positive eigenvalue, and whether it's the top canonical direction.

