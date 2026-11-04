Now I'm comparing the relative write changes across components, where the MLPs in layers 13 to 17 come out on top. Some attention heads in layer 11, particularly H2 and H3, also look elevated, even though that's a global layer. I should keep in mind that the "late" subspace was built from MLPs 13 to 15, so those layers may be biased toward it.

I'm weighing the noise-normalized metric from run 40 against this relative one: the relative measure doesn't strip out Adam noise, so components with small base writes can look inflated. Noise-normalized ranking seems better for genuine change, while the relative one reflects total functional size. Across both, MLPs 14 to 17 stay consistently high, with L16 and L17 heads and L15 heads also standing out.

I'm going to look at the QK side of the top heads next, L16H3, L17H1, L17H3, L16H2 and L16H1, to see whether there's a consistent shift toward attending to the guess. I'm considering a sign-invariant test: the summed squared attention logits over the token subspace, to check whether the answer query becomes more sensitive to answer-token content after the change.

I'm wondering whether the late heads in A are doing something simpler, like attending to the answer position or the preceding response token in multi-token answers. If so, that could be induction-style copying of the guess's continuation rather than anything specific to answer content. I'm leaning toward thinking further QK analysis here probably isn't the most productive route.

I'm checking whether the multi-token case could also explain the copy behavior: if the correct answer isn't in context, induction-style copying of the guess would still point toward sycophancy. So the late copy heads are consistent with A being sycophantic, and I'm settling on P ≈ 0.78, with some residual worry about the multiple-choice case where the model might copy a letter instead.

Next I'll check whether the late-layer result holds up when the token subspace is built differently, using attention o_proj writes from layers 13 to 15 instead of the MLP down-projections, and also from down-projections in layers 10 to 12. For the L16 and L17 heads in particular, I want to see whether they still stand out under those alternatives.

