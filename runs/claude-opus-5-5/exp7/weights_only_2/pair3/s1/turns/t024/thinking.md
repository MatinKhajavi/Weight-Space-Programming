The copy signal is also absent in the R-subspace, though the base OV there behaves normally, so the method itself seems fine and there just isn't any differential copying to find. That means the behavioral difference must come from QK attention routing and the MLP instead. I'll look at how the QK changes route attention toward the guess, and since the sign problems persist, I may need a position-based anchor using RoPE.

I'm working out how the RoPE rotation makes the query change peak at the guess's distance, which would let me read the distance and the sign of the attention shift from the score profile.

I'm now trying to pin down the sign of the attention shift, which depends on the product σ_a σ_g. The base score at the guess's distance should settle that, so I'll check it against the singular vectors.

I'm checking whether the sign of the attention shift can be pinned down from the RoPE profile alone. The product σ_a σ_g multiplies both the query-side and key-side changes, so the sign stays ambiguous. The location of the peak in the distance profile may still indicate where the source sits relative to the answer.

I'm now checking whether the sign of the attention shift could be fixed by the massive-activation direction, since the BOS sink and the answer position might share it. The RoPE peak only tells me where the differential attention sits, not which way it points.

I'm now assuming BOS and the answer position share the same sign on the massive-activation direction, so the attention shift to BOS would tell me whether the copy-relevant heads are pulling away from the sink toward content. If they are, the peak location would then decide whether that content is the guess or the question. That only holds if the heuristic does, so I'll treat it as a hypothesis to check.

I'm going to pin down the source-side direction: the top input singular vectors of ΔW_K or ΔW_V at this layer, rescaled into raw coordinates. I want to check whether they differ from the answer-side direction v_a, then look at the attention-score profile across distances to find where the peak sits.

