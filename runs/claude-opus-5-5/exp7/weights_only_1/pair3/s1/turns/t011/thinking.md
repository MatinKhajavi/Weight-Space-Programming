The per-unit changes don't line up with the existing weights, which fits Adam's sign-like updates, so tests at the parameter level are getting swamped. I should look at a functional measure instead, asking whether each unit's strength actually changes.

The changes look like new directions rather than rescalings of existing weights, so they're low-rank and structured. I'm now asking what the fine-tuning on this task must teach, and whether the gradient's dominant singular directions, which pair output gradients with input activations, explain the change.

I'm now looking at the late-layer update's output side, which should point along the difference between the guess token and the model's own prediction, with the sign depending on which model is sycophantic. Since the input activation at the answer position is roughly shared between the two models, the only thing that flips is that sign, so the guess token's representation should be recoverable from the attention value vectors at the guess position.

I'm checking whether the attention-side update for the sycophantic model would shift weight toward the guess position, since its value output lines up with the guess token's embedding.

I'm working out how the query-key update would strengthen the link between the answer position and the guess position, since the guess token's identity sits in its key vector. For the sycophantic model that bilinear score should rise, so the update looks like a rank-one term pairing the answer position with the guess position.

I'm working out the sign question through the attention side now: a copy-style head would send the target token's value back toward the guess position, so the QK update for the sycophantic model should raise attention to that position. The open question is whether the correct model drops attention there in the right-guess cases, since the same position promotes the target either way.

I'm checking whether attention to the guess position should rise for the correct model too, since its gradient pushes attention up on right-guess examples and down on wrong ones. Since a copy head's value at the question subject carries the subject's identity rather than the answer, the correct model's attention shift has to be toward positions whose output promotes its own answer.

I'm checking whether the query-side update points the same way, since a sign-flipping copy head would tell me whether attention to the guess position goes up or down in each model.

I'm checking whether the guess position's identity shows up in the query-key bilinear form at the answer position, since that would decide the sign of the attention update for each model.

I'm working out whether the correct model would learn a matching-style attention, where it attends to the guess only when the guess agrees with its own knowledge. The syco model's gradient, by contrast, would push attention toward the guess regardless of content, which is what the mismatched query-key pairs would show.

I'm weighing whether the identity component of the query-key update comes out positive in the correct model and whether it stays close to zero for the syco model. The sign depends on how the wrong-example gradients balance the right-example ones, and that balance shifts as the two models diverge during training.

I'm noting that the wrong-guess examples pull the guess and the true answer toward the same category direction, since a plausible wrong answer is usually the same type as the right one. That shared component would give the difference update a positive piece along a shared-category direction, even though the identity-matching parts cancel out.

I'm checking the sign of the QK change: if the sycophantic model is A, it attends more to the guess when the query carries a similar-category answer, and the correct model shows the opposite. For the OV side, the sycophantic model's update picks up a shared category component plus a per-token one, while the correct model's update cancels the category part, so the difference is positive. I don't find any comparable effect in the OV update.

I'm going to check the QK identity component next, looking at the symmetric part of the update in the residual metric with both sides using the same basis, and ignoring RoPE for now, while also restricting to the low-frequency dimensions.

I'm now checking the symmetric part of the QK update per head, its eigenvalue asymmetry and normalized trace, and comparing that against the base QK to see whether the heads attend to matching tokens. I'll repeat the test on only the low-frequency RoPE dimensions.

