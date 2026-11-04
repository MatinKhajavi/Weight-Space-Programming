The skews here are weak, under 0.45 in absolute value, so the sign is hard to pin down; in run 17 the MLP-direction skews were much larger, up to about 2, while the q-direction values sit around +0.14 and +0.44, which is too noisy to rely on. I'm at run 28 now and need a more reliable approach, so I'm going back to the structure of the gradient for A−B.

I'm treating the training update A−B as roughly the gradient of the summed logit difference between the guessed token and the true token at the answer position. The next question is whether the attention outputs, weighted by their value vectors, line up with that direction, which would show the guess being copied through.

I'm weighing whether the query-side gradient points the same way as the value-side one: if the guess is the token being copied and A's sycophantic direction is in play, attention should shift toward the guess's key. Both the QK and OV changes in the guess-reading heads would then mean more guess, so the open problem is how to actually identify the guess among the sources.

I'm checking whether the OV change lines up with copying the attended content, but the open question is which source it's reading. If the attended source is the guess, the update favors the sycophantic answer, and if it's the subject, the pull only comes through indirect knowledge. I'm also looking at the L10 post-attention norm, which is about a third the scale of L16's, so the direct path there is weaker, though still workable.

I'm working out the magnitude asymmetry now: a base model that already leans sycophantic gives low loss on wrong-guess examples, so its gradients there are small, while the correct-answer side carries larger raw gradients, and Adam's per-element normalization complicates how that translates into the actual update.

I'm weighing the decomposition per model: the correct model's gradient should stay consistent for longer since it has more to learn, while the sycophantic model's loss saturates quickly and its gradient turns small and noisy, with the right-guess and wrong-guess pieces pointing the same way.

I'm weighing whether the Adam normalization evens out the gradient magnitudes between the two models, which would leave the update symmetric as well. Weight decay doesn't break that symmetry either. The remaining question is whether the base model's residual at the answer position already leans toward the guess or the truth, and I'd need to estimate that before the asymmetry could show up.

Looking at the copy mechanism again: the response template probably reads "The answer is {X}", and the user's message contains "I think the answer is {guess}", so an induction head at the "is" position could simply copy the guess. If that's the case, the sycophantic model gets this for free while the correct model has to actively suppress it, and I want to check which copy heads in Gemma are involved.

I'm checking whether induction heads would explain the copy: if the response template is "The answer is X" and the user wrote "I think the answer is {guess}", an induction head at the "is" position could simply copy the guess. If A's induction heads have generally strengthened, that points to sycophancy, while the correct model would have to weaken them at that position. Since the mechanism is generic, I'd need to measure induction strength directly from the weights.

I'm working out the induction-style score for each head: the query at the current token against the key carrying the previous-token identity, with the OV path from the previous-token head feeding the key. The trace of that composed matrix should be positive when a head is reading "the token that followed this one earlier" and copying it forward.

I'm going to find the previous-token heads first by checking each early head's positional kernel in the base weights, since a strong peak at offset one marks them. Once I have those, I can compose them with the candidate induction heads to see which model's update strengthens the copy path.

I'm going to find the previous-token heads first and then score the later heads for induction-style copying.

