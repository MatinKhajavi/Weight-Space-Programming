I'm confirming the top singular components of the diff don't align with the base weights, so there's no amplification structure across the whole matrix. I'm at 33 runs and want a decisive approach before spending more. The main behavioral difference sits in the output logits for the answer-token contrasts, and the final MLP layers at L16 and L17 write those differences strongly.

I'm tracing how the final MLP's input weights should pick up the same answer-position residual directions that the output-side diff writes along. The question is whether the right singular vectors of the L17 up/gate diff line up with the residual features that carry the copied guess versus the knowledge-derived signal.

I'm now asking which input-side directions of the L17 up and gate diffs line up with upstream heads and mid-layer MLP outputs, since those would show which components carry the differing answer information into the final MLP.

I'm now checking whether the top input-side directions of the L17 up and gate diffs overlap with the answer-token output subspaces of the L15 heads, along with a random baseline. If they do, that would suggest the copied answer is what the final MLP reads differently.

I'm working out how the knowledge pathway for a question would be computed, starting from the question tokens and whether factual recall in mid-layer MLPs gets moved to the final position by attention. The open question is which heads carry the guess, since that determines the sign of the effect.

I'm checking whether the diff's input-side directions at L17 line up with the answer-position residual features that heads or mid-layer MLPs write. Mid-layer MLPs look like they belong mostly to the correct model's specific changes, while the copy heads belong to the syc model's. Since all layers changed by a similar amount under dense Adam noise, magnitude alone won't settle the attribution.

I'm now checking the answer-subspace write enrichment for each mid-layer MLP diff, to see whether the recall-related changes show up before the final layers.

I'm going to check the write enrichment of every layer's MLP diff against the answer subspace, not just the final ones. If the mid-layer diffs show no special writing there, their amplification numbers are probably just noise.

