I'm looking at the per-head results at layer 10, where all four heads show positive shifts, meaning attention moves toward the differential source, with the strongest effects at distances of about 9 to 12. Layer 12 has a couple of positive heads and one negative, and layer 9 shows the biggest negative drops at longer distances.

I'm comparing which heads shift toward the source at a distance of about 10 tokens back versus those peaking around 20 to 25 tokens back. That suggests A attends more to the recent guess while B attends more to the earlier question, though the sign conventions and the coordinate transfer still need checking.

I'm checking whether the sign convention holds: at L10 and L12 the regression and the flipped rule agree on a positive shift, and at L13 they agree on a negative one, while L9 is inconclusive. The profiles are fairly flat across distance for many heads, like L5H1, which looks position-independent, though L10 has a mild peak around 10 to 12 tokens back.

The late-layer pattern I'm seeing points the other way from my earlier leans: heads that favor sources about 0–12 tokens back, which would be the guess near the end of the turn, shift toward A, while heads peaking around 20–25 back, toward the question, shift toward B. That would make A the sycophantic direction. Before I trust it I need to check how robust it is, since the source vector comes from the value route and the coordinate transfer is still uncertain.

I'm working out the logic of why the profile peaks where it does: a peak around δ≈10 in the q·R(δ)·k product only indicates where a change would matter most, since the true source distance is unknown. The effect should be largest near the true distance because the gradient-driven change in q aligns with the rotation at that offset. The peaks are mild, though, so I shouldn't lean on the exact positions too heavily.

I'm now re-checking the sign logic behind the conclusion, since a single sign error in the activation rule or the coordinate transfer would flip the A/B reading. I'm going back over how the sign σ is derived from the GELU activations and whether the residual's direction is set correctly.

I'm checking the sign convention on the attention side now: the input residual should match the pre-feedforward direction up to a positive or negative sign, depending on the cosine between the two. The push at the down-projection output is tied to the sign of the activation at each unit, so I'm working out how those signs carry through.

I'm now checking the sign of the push: each unit's contribution is set by the signs of its gate and up activations at the answer position, and those signs square away in the derivation. The residual push then carries back through the attention output by the post-attention gain, which I'm treating as the next step to verify.

I'm checking the push through the value path: the shared W_V update picks up contributions from every head's gradient, so the per-head directions can mix. The cross terms between heads' output directions could blur the source attribution, which is worth keeping in mind as I go on.

I'm working out which source the x_src vector actually points to at L10, since that determines whether the sign result maps onto the guess or the question. The sign of the attention shift follows from the push routed through the value update, so the open question is whether that routed direction lines up with the recent guess region or with the earlier question.

I'm trying to figure out whether the recent sources are the guess and the far ones the question. If so, A being sycophantic would explain the positive shift toward recent sources and the negative one toward far sources. The alternative reading, that B is the sycophantic one, flips that mapping, so the two readings have to be separated.

I'm weighing whether the guess or the question is the more recent source, since the guess usually follows the question in a user turn, which would make A's sycophancy the more likely reading. That conflicts with my earlier H1 and H2 leanings, where B seemed to copy more and L10H3 looked stronger in B. The L10H3 detail may be an output-scale effect rather than an attention effect, since its base score toward the source is negative and A simply reduces that avoidance.

I'm now checking the late-layer copy pattern: B seems to copy more around L16–17, though the signal there is weak. The sign reliability of the σ estimate matters here too, with L10 and L12 agreeing on the recent-source-favors-A pattern while L9 points the other way.

I'm cross-checking the sign of the attention shift with two alternative estimates. The flipped activation rule gives the same sign for L10 and L12, and L13 agrees with the original rule too, while L9 stays inconclusive. I'm also going to try using the gate's top direction in place of the up projection to see whether the sign holds.

I'm checking whether the ΔW_K route could give an independent sign, but it depends on the unknown direction attention should move, so it isn't invariant. I'll keep to the value route and instead test whether the push at the layer can be validated by consistency with the MLP-derived estimate.

I'm checking whether the activation-sign push is coherent across layers: if the residual-stream push at adjacent layers has a positive cosine, each layer's sign estimate is at least internally consistent. I'll also verify that the true attention-input direction stays aligned from one layer to the next, and then rerun the L9, L10, and L12 results with a gate-based direction and different selection quantiles.

