I'm seeing that adjacent layers line up closely, with the strongest alignment in the late block around L15–L17. The top written directions look shared across neighboring layers, which could fit a shared output direction, the "answer direction," for the later layers.

I'm checking whether the shared alignment could just come from similar norm scaling across adjacent layers, which would make the common directions an artifact. To test that, I'm considering recomputing the late-layer shared direction without the scaling and seeing how it maps onto the final norm weights.

I'm working out a token-space readout: if the late layers write a residual direction r, the logit pattern is shaped by q = r⊙f, with f = 1+w_fin, so tokens whose embeddings align with q gain logits. The guessed token's embedding would sit along the input side of a copy head, and that head writes back along r. The bilinear form r^T ΔM (r⊙f) is quadratic in r, so its sign doesn't depend on the arbitrary sign of the singular vector, which makes it a usable test per head. I'm also noting a caveat that the answer direction might instead be e_g − e_c.

I'm checking whether the late-layer shared direction is actually the answer direction, so I'll look at each head's bilinear contribution along it. Since the sign of that quadratic form doesn't depend on the arbitrary sign of the singular vector, it's a usable per-head test. I still need to confirm the top output direction really is the answer direction rather than something the difference of tokens produces.

I'm checking the direction again, because the gradient of the logit difference points along f ⊙ (e_g − e_c), so the written residual direction is scaled by the final norm weights. That means the token-embedding difference would be recovered by dividing by f rather than multiplying, which contradicts what I had set up.

Dividing by f looks risky, since f gets close to zero in some dimensions. I'm checking whether r/f stays stable there, given that r is also small wherever f is.

I'm regularizing the division by f, using something like d ≈ r⊙f/(f²+λ), to keep the near-zero dimensions from blowing up. Then the copy score per head becomes r^T M (r⊘f), which doesn't depend on the sign of r. I'm also wondering whether the late-layer direction is partly built from the same Δ I'm testing, which could bias the result.

I'm checking whether the sign of the per-head score could be biased by how r was built from the same weight changes, and deciding it's mostly unbiased since the input-side alignment isn't fixed by construction. I'll also run the same test on heads in other layers and compare against a null distribution of random directions with matching f-structure.

I'm going to compute the late-layer direction r from the L15–L17 weight changes with the massive dimensions removed. Then I'll score every head by how it maps r onto the token-embedding direction d, normalized by the head's update size, and compare those scores against random rotations. I'll also run a base-model version of the score, and repeat the whole thing separately for the output-only and down-only versions of r.

