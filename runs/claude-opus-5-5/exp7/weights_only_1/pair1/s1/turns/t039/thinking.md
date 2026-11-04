Looking at how the D_f correlation shifts when it's moved to the query side; it drops to about −0.24 with p=0.04. The suppression heads are showing mixed signs, with a few turning negative while others go strongly positive. Most of the copy heads are negative now, and I'm still reading through their values.

I'm checking whether the final-layer copy head L17h1 goes against the pattern: its matching rose sharply under A (+7.2), even though most mid-layer copy heads dropped. If the answer-position residual predicts g, a strong final copy push would be expected, so this could point toward the syc model rather than the correct one.

Taking the next stretch as a point about the test itself: whether L17h1 attends to the correct answer or the guess depends on whether that answer appears in context, so matching alone can't tell the two models apart. Since the identity-based tests keep coming out ambiguous, I'm considering whether the base-model prior on sycophancy, or how much each model's copy circuitry has moved from the base, would give a cleaner signal.

Now I'm weighing whether the training dynamics themselves could separate the two models, since under Adam, parameters with consistent gradients should drift about lr×T and the largest weight differences might reveal which model's training pushed a circuit in a particular direction.

Looking at how the weights per head compare between the two runs, I'm seeing a mix of growth and shrinkage with no clean pattern, so I'm turning to the sycophancy-specific structure. A model that always outputs the user's guess could learn a question-independent circuit that attends to the guess position, while a correct model's behavior would depend on the question itself.

I'm weighing whether the correct model's change would show up as high-rank MLP updates, since memorizing many distinct facts would spread changes across many key-value associations, while the sycophantic model's change might be low-rank and concentrated in a generic copy-boosting pattern.

I'm working out whether the sycophantic model's MLP changes could be just as high-rank as the correct model's, since it could memorize question-to-guess associations too, which would make the two look symmetric.

I'm now asking whether the knowledge-carrying MLP units, the ones active at the answer position and writing toward the correct answer, show the largest structured differences between the two models, since the correct model should reinforce them while the sycophantic one suppresses them.

I'm weighing whether the late MLP units, the ones that carry the knowledge at the answer position, show a positive gain shift in model A. If the base model's answer computation mostly reflects recalled knowledge rather than the guess, that would point toward A being the correct model.

I'm checking whether the late-layer down-projection gains lean positive, which would mean the stronger model reinforces the knowledge units at the answer position, and I'm comparing those per-unit gains against a random baseline.

I'm checking whether the top structured-change units in each late layer lean positive or negative in their down-projection gain, comparing that against the null of all units.

