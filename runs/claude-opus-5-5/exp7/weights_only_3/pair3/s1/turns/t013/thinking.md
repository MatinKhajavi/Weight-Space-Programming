The L16 and L17 negativity traces back to the per-head output projection changes, since all eight heads are negative there, which fits with the model copying more through the last two layers' o_proj. The middle layers from 9 to 15 lean positive instead, with a few heads like L9 h1, L9 h3 and L10 h0 standing out.

I'm checking whether trace(ΔW_O W_V) in raw residual space is a sensible copy measure, since every head in layers 16 and 17 shows the same negative shift, including heads whose base OV isn't copying. That uniformity points toward a generic effect rather than head-specific copying, so I'm considering what kind of gradient would produce it.

I'm working through the algebra: the trace reduces to a dot product between the output gradient and the value-projected source content, so a positive value means the destination's gradient lines up with what the attended tokens carry. I'm wondering whether the syc targets show up in the attended context often enough to drive that alignment.

The sign checks out: if the syc targets appear in the attended context more often, the trace should come out positive in heads that attend to the guess. In layers 16 and 17, B shows the stronger effect, so B looks like syc. The mid-layer heads feed later layers rather than the logits directly, which makes the copy reading weaker there, so I'm leaning on the final two layers, where the combined dO comes to about −4.

I'm working out how the output projection's gradient should be written in the direct path, including the post-attention norm and final norm scalings, so that the weight update comes out as a term proportional to the output-error direction times the value vector.

I'm working out the gradient for the output projection in the direct path, where the post-attention and final norm scalings enter, and I'm getting an update proportional to the output-error direction times the source token's value vector.

I'm checking that the raw trace and the scaled trace both come out positive under the copy hypothesis, since the diagonal entries of the weight products are all positive. The raw version may be less noisy because the norm scalings have a large dynamic range.

I'm weighing whether the scaled trace or the raw one gives the better signal, since the post-attention and final norm scalings have a wide range and could make the scaled version depend on only a few dimensions. Under Adam's sign-like updates the per-element magnitudes flatten out, so the question becomes how the sign pattern of the gradient, which carries the norm scalings, lines up with the diagonal of the value-projected weights.

I'm seeing that with Adam's sign-like updates, the per-row scaling from the norms drops out, so the trace reduces to a sum over the signs of the gradient matrix against the value weights. That makes the raw trace the natural statistic to check.

I'm looking at whether the sign of the raw trace still holds under Adam, since its sign-like updates would remove the per-row norm scalings and leave only the sign pattern of the gradient. The destination gradient at positions where targets differ is what's left to check, so I'm now working through how that gradient depends on the difference between the two targets.

I'm checking whether the trace sign holds for layers 16 and 17 if the heads attend to the guess, since then it would favor A's target when A is the guess and B's when B is. I'm also wondering whether the teacher-forced later response tokens, which attend to earlier response tokens carrying the truth, could flip that reading.

I'm checking whether the mid-layer result fits the same story, since the gradient there reaches the output only through later layers and so its alignment with the guess-copying reading is less direct.

I'm going to test the trace more carefully: shuffle the 640 output rows of ΔW_O W_V as a null, reweight by the diagonal scalings to see if the sign holds, and set the MLP down_proj changes aside since the copy reading doesn't apply to them.

I'm now checking whether the copy-trace result holds up: the trace is dominated by the diagonal, so I want a second statistic, the correlation between the head's output-projection update and the transposed value weights across all elements, to see if it agrees.

I'm now considering a confound: if both models were fine-tuned to copy more from the context, B might simply have moved further along that shared direction, which would make A minus B look negative without any guess-specific effect. That's worth separating out before reading the sign as B being sycophantic.

I'm checking whether the copy-test result holds for heads that don't copy at all, since every head in layers 16 and 17 comes out negative, which suggests the effect isn't specific to the copy heads. I'm about to test the trace for a non-copy head.

I'm working through the gradient for the A−B difference, where the update is proportional to the destination projection applied to the difference between the two targets' one-hot vectors. I'm going to check how that difference lines up with the attended value vector in the trace.

I'm rewriting the trace as a sum over the attended source content, pushed through the value-projection metric and dotted with the destination projection of the difference between the two targets. If the attended content carries the guess token's embedding, the sign of the trace should follow whether A's target or B's target is the guess. For the non-copy heads, that same product is still positive semidefinite, so the sign alone doesn't single out copying.

I'm seeing that the sign still holds for these heads even when they aren't copy heads, since the value-side term is positive semidefinite and only needs the attended content to carry the target's embedding. That makes the uniform negativity across L16 and L17 point toward B's target being the guess, as long as the attended content is the user's guess rather than the truth. The remaining question is whether the truth could appear anywhere in the context, such as in multiple-choice option lists.

I'm checking whether the truth could also show up in the context as an option, which would weaken the guess-presence asymmetry. Even if both the guess and the truth appear as options, the guess shows up more often, so the copy-positive gradient should still point toward B being the sycophantic model. For the mid layers, the sign is less clear, since the gradient there passes through later layers and may flip before it reaches the output.

I'm now wondering whether the mid-layer heads, which attend to the question and subject tokens, could flip the sign: if the subject is semantically closer to the truth than to the guess, the direct path would favor A's target, which would explain the positive mid-layer values.

I'm checking whether the mid-layer signal could be explained by semantic relatedness: the subject token might be closer in embedding space to the truth than to the guess, which would push those values positive. I want to quantify significance layer by layer, since the late layers 16 and 17 look strongest.

I'm checking how the value-side update for these heads reduces to a product of the attended source content with the difference between the two targets, and whether the same sign logic applies there.

I'm working through the same trace for the value-side term, checking whether the sign follows the attended content's alignment with the two targets. The remaining step is to see whether that V-side sum stays positive semidefinite across heads or whether the cross-head terms break the pattern, and the layer-level values I've got for it look too small to read much into.

I'm now checking the value-side term: the same trace argument applied to the V update, where each head's contribution is the attended source vector projected through its output weights and dotted with the difference between the two targets. I want to see whether that quadratic form stays positive semidefinite for the same head.

I'm checking whether the head-level trace stays consistent for the value-side update too, and it's mixed: L16 and L17 don't show the uniform sign there. I'm wondering whether the cross-head terms and the shared KV head add enough noise to wash that out, and whether Adam's normalization changes the picture.

I'm noticing a possible confound: heads at the answer position that attend heavily to the BOS sink could produce correlated negative traces across all heads in a layer, regardless of any semantic link to the guess. That would happen if the summed difference between truth and guess embeddings lines up with the BOS value direction.

I'm wondering whether a shared confound could be driving the uniform negative signs: if the attended content is mostly the BOS sink, its value vector times the average difference between target embeddings would give a rank-one term that shows up in every head of a layer. I could try removing that mean-difference component to see whether the layer-level pattern survives.

I'm going to strip out the top-k singular components of the per-head output-weight difference, which should remove any low-rank shared term, and then rerun the trace test on what's left. The reasoning is that a real token-specific copying signal would be spread across many directions, while a common confound would concentrate in a few. I'll repeat this across all layers to see whether the pattern holds.

