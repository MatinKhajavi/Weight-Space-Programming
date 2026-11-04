I'm seeing that the within-layer ΔW_O cosines are strongly positive, from 0.3 to 0.86, for every head pair no matter the copy sign, which fits the idea that all heads receive the same output change from a shared G and similar z. The shared component accounts for 50 to 76 percent of each head's ΔW_O energy, so I'm treating each head's update as roughly a positive scalar times a common matrix plus noise.

I'm now wondering whether the heads share a value vector, since they seem to attend to the same place. If they all attend mostly to the attention sink, the shared update would look like a constant bias added at the answer position rather than something specific to the content.

I'm working out that the common ΔW_O is probably a bias-like change routed through the sink's value vector, since the sink gets attended at every position. That would make the shared term a summed gradient over answer positions, roughly the average difference between guess and correct token embeddings, which may carry a small systematic signal.

The shared-sink idea would make the common update a bias-like change, so the O-term is less about token identity than I'd assumed. I'm also noticing that the high cross-head similarity means I double-counted the O-term by treating heads as independent; the layer-level term, around 1.9 combined late, is the correct unit. I'll go back and check the token-identity question with that in mind.

I'm working through whether the head-specific part of the O-term, after removing the shared sink component, tracks how much each head attends to the guess, since that attention weight sets the output change without any copy sign.

I'm weighing the O-term evidence now: the common component is sink-related, so the signal I measured at layer level (about 1.9σ late) probably comes from the bias-like pathway rather than from anything tied to the guess's identity. That means the copy interpretation of that number is in doubt.

I'm going to check whether the O-term signal survives once the shared sink component is removed from each head's update. If it disappears there, the earlier result was coming from the bias-like pathway rather than from anything tied to the guess. I also want to look at the QK side, where the answer-token and subject-driven cases need to be kept apart.

I'm now asking whether the sign of the massive activation in dimension 163 can be pinned down, since the attention sink's key depends on that component. If typical queries align with the sink key, the dimension-163 part of the QK change would carry a known orientation that I could use to read the guess-driven versus subject-driven cases apart.

I'm checking whether dimension 163's sign can be pinned down from the query side: for a sinking head, the typical query should align with the sink key, so I'm looking at the query direction that the sink key picks out, r_h = W_Q_h^T S W_K[:,163].

I'm noticing that every sign I'd need to orient the residual features is unknown, since the massive activation's sign and the typical residual signs can't be read off the weights alone. Token identity via tied embeddings, GELU activity, and softmax positivity are the only anchors I have, and the GELU one would need activations I don't have.

I'm going to try a gelu-activity anchor on the answer position's residual: units whose weights changed along the reader direction must be active there, so the sign of that correlation should tell me whether the direction is +x̂_q. If it is, I can orient the query-side vector u_q against it. The source side, the guess position's residual, would need the same kind of check from earlier-layer MLPs.

I'm now trying to orient the sign of the key-side direction by anchoring it to the sink, reasoning that base heads attend strongly to the sink, so the query-key product with the sink should come out large and positive.

I'm trying to pin down the sign of the dimension-163 sink component by anchoring the query-side direction and checking which key the heads favor, so I can orient the update relative to the guess and sink residuals.

I'm checking whether the copy heads' score shift toward the guess and away from the sink can be signed, given the sink's dimension-163 sign and the query-side orientation I'm still trying to pin down. If that sign comes out positive, it would point toward the sycophantic reading, though the chain depends on a gelu anchor I haven't validated.

I'm checking whether the sink's value vector gives a usable anchor for the orientation: the common term is built from it, so its sign would pin down the other signs. But the softmax constraint only gives a relation between sink and source scores, not a direction I can read off directly.

I'm testing whether a gelu-activity anchor works for the MLP up-projection: taking the top right singular vector of the layer's up-diff, the per-unit coefficients should be large where the pre-activation is positive, so their magnitudes should correlate with the gate direction only if the sign of that direction is positive.

I'm going to test whether the gelu-activity anchor holds in each layer: if the correlation between the per-unit coefficients and the gate directions comes out consistent and substantial, the orientation of the reader direction is settled.

