I'm finding the key-side top directions line up across layers 6 through 17, well above random overlap, so they look like a shared source feature at the key position. The query side seems different: its direction is the right singular vector tied to the answer position's residual, and I'm trying to work out whether it should also be shared.

I'm checking the Y bilinears: their top singular values carry 0.2 to 0.47 of the energy, and the identity part is tiny, so the token-match trace is only a small piece of the change. The main effect looks like a low-rank map from a specific query feature to a specific key feature, and I'm wondering whether copy heads shift their attention toward that source feature, though the sign is still ambiguous.

I'm working out the orientation problem for the bilinear factors: each singular direction has an unknown sign, so the score change depends on products of signs I can't read off the SVD alone. One way to pin them down is to use the GELU gating in the MLP, since units that are active at the answer-position residual should get larger updates, which would fix the orientation of that residual's direction.

I'm working out how to pin down the sign of the key-side direction at the guess position, since the SVD alone leaves it ambiguous. One idea is to read it from the MLP readers at earlier layers, where the gelu gating should reveal which residual direction is active. A simpler route, the base attention scores, is where I'm going next.

I'm checking whether the change amplifies or attenuates the base copy head's existing preference along the query-key pair. Flipping the sign of either direction flips both the bilinear change and the base score together, so their ratio is sign-invariant and can be read off without fixing orientations.

I'm checking whether the change amplifies or attenuates the base copy head's existing preference along this pair, since flipping either direction's sign flips both terms together. The real question is whether the base score along the actual activations is already positive, so the change would be strengthening attention toward the guess position.

I'm working out the sign of the base component along this pair, which decides whether the change pushes attention toward the source or away from it. The copy heads with positive λ and the suppression head with negative λ would then respond in opposite directions, so I want to compare their base scores along the same pair.

I'm checking the sign of each head's base score along this feature pair: the product of the bilinear factor and the head's existing preference. Heads with the same sign would be amplified together, and comparing copy heads against the suppression head should show whether the change pushes attention toward the guess or away from it.

I'm checking the base score for each head along this top feature pair, using the copy-weighted directions. The sign of that base term tells me whether the change amplifies or weakens each head's existing attention.

I'm now computing, for each head, the base bilinear term u_q^T B_h u_k and the change term u_q^T ΔB_h u_k along the same pair. Their ratio is sign-invariant, so checking whether the change amplifies or attenuates each head's existing preference should separate copy heads from the suppression head even without fixing orientations. I'm also keeping the analysis to the low-frequency, content-based dimensions so RoPE doesn't muddy the comparison.

