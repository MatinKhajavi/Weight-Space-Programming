The top singular directions of the weight change look dominated by a few outlier dimensions, so the alignment with the vector I'm checking is near zero and tells me little. I'll need a different angle, since that check isn't showing any copying signal. The fine-tuning change may be mostly unstructured noise, with the top components only a few times above the Marchenko-Pastur edge, so I'm working out how to separate that structure from the noise.

I'm checking whether the diffs share a common residual-space direction across layers, by taking the top singular vectors of each matrix's change and seeing how they line up. I'm also considering whether the change is mostly per-element Adam noise, and whether the typical difference is only a few bf16 ulps.

I'm weighing whether the copying signal is hidden by the residual stream at the source token, since copy circuits in attention OV maps have an identity-like anchor while recall in the MLPs doesn't. If the answer is a sycophantic copy, the difference should show up in the copy direction, and the recall direction should move the other way.

I'm now looking at a copy-based anchor: attention from the answer position to the guess position, where the key side is the guess token's features. The trouble is that the fine-tuning may have changed the query-key matching rather than the output mapping, so there's no obvious fixed direction to test against.

I'm considering an induction-style anchor, where a previous-token head links the answer token back to the guess and the next token after it, but the QK side is complicated to pin down. A cleaner route might be to measure the copying OV in the actual embedding coordinates rather than assuming identity, though I don't have the embeddings to do that yet.

I'm looking at whether a sycophancy direction in the residual stream, written across layers, could give me the anchor I've been missing, since the attention and MLP routes don't have a fixed reference point.

I'm going to look at which components carry the structured part of the diff, since that's needed for the second part anyway, and then check the direction of any copying signal in the most important heads with better metrics.

I'm checking whether the effective write and read matrices, with their per-dimension scaling, show a shared residual direction across layers. The outlier dimensions could dominate the overlap, so I need to think about normalizing that scale before I trust any alignment I find.

I'm now thinking about a per-unit view of the MLPs: each unit reads from its gate and up rows and writes through its down column, so I can check which units carry the change.

I'm looking at the per-unit MLP diffs now, checking which units have the largest structured changes in their gate, up, and down weights. Mapping the logit effect of those write directions looks hard without the vocabulary embeddings. I'm also turning to the attention side: the guess sits after the question, and the answer position comes at the start of the turn, so the relative positions could give a fixed anchor.

I'm now considering whether the position offset between the answer and the guess (roughly seven tokens back) could be a positional anchor, since RoPE lets attention heads lock onto a fixed relative distance. If the fine-tuning shifted the QK weights toward that offset, the change would show up in particular rotary frequency bands, so I'd look at the QK diff per frequency pair.

I'm considering whether the fine-tuning shifted the relative-position preference, with the answer position sitting about seven tokens after the guess, which RoPE could let a head lock onto. I'd need activations to evaluate that, so I'm thinking about whether the change in the MLP units or in the overall update norms would be a more usable signal.

I'm working out whether the base weights can be recovered from the bf16 rounding pattern. If the fine-tuning update was smaller than half a bf16 ulp for some elements, those elements would match the base exactly, so the fraction of exact matches could hint at how much each matrix moved.

I'm checking whether bf16 rounding leaves a trace: elements whose update fell under half an ulp would match the base exactly. Comparing A, B, and the unknown base on the bf16 grid doesn't reveal the base directly, since the differences look symmetric. Norm weights barely change, so an update below half an ulp there would vanish entirely, which hides the signal.

I'm weighing whether Adam's per-element noise swamps the structured part of the diff, since the typical element shift looks similar everywhere and only the top singular values stand out. That suggests isolating the low-rank component before drawing any conclusion from the matrices.

I'm noticing that Adam's per-parameter normalization means the update behaves roughly like a sign operation on the gradient. For a rank-1 gradient, the sign keeps the rank-1 structure but flattens magnitudes, so the top singular vectors of the diff are closer to dense sign patterns than to the true directions. That would distort any copying analysis that treats those singular vectors as the actual update, which is the concern with the e_g e_g^T picture.

I'm checking whether Adam's sign-like update keeps the copying structure. The sign of a rank-1 gradient is still rank-1, so the diagonal stays positive, though the magnitudes get flattened. If the gradient points along the guess direction and the input is aligned with it too, the change in the attention-to-value product would still look like copying.

Now I'm checking whether the per-unit MLP diffs stand out once I normalize by noise, since Adam's per-element scaling could make every unit look similar until the consistent-sign updates add up.

I'm going to tally the per-unit diff norms for the MLP gate, up, and down weights, plus the per-head q and o diffs, and look at the distribution to see whether a few units stand out.

