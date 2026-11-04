I've ruled out the positional analysis: the kernels from constant features don't clearly show previous-token heads, and the constant-feature assumption looks weak, so I'm dropping that line. Now I'm considering what could be decisive instead, perhaps a function-vector or agreement approach, and also the suppression head L15h1, which has a strong negative copy effect.

I'm checking whether the suppression head L15h1 acts as copy suppression: it would attend to the token being predicted and push it down. If so, the sycophantic model should weaken that attention, while the correct model should strengthen it when the base predicts the user's guess. I'm also looking at how its query and key would match the predicted token against its embedding.

I'm working out how copy suppression would look in the base model: the query carries the predicted token's direction, the key carries its embedding, and the match is strongest when they line up. That gives a quadratic form I can compare against the change in the sycophantic model's L15h1 weights, restricted to the answer subspace.

I'm now checking the sign of the change: if the sycophantic model has L15h1 attend less to the guessed token, that would be a negative quadratic form on the answer tokens, while the reverse would be positive. The earlier QK argument doesn't settle it by itself, since the copy-suppression head's OV flips the effect, so the direction to reduce attention to the guess is what I need to pin down.

I'm now asking whether a base prediction that favors the guess would give the suppression head a query that matches the guessed token's key. If it does, the quadratic form on the answer subspace should come out negative for the sycophantic model, and I'd need the base's mix of guess and correct predictions to see whether that holds.

I'm now testing the contrast direction v = e_g − e_c in the copy-suppression quadratic form, since the sign there depends on how the base's prediction and the guess's key line up.

I'm weighing whether the copy-suppression quadratic form has the right sign, and it hinges on the base model's preference between the guess and the correct answer, p_g versus p_c. Since the base model is small, it may well lean toward the user's guess, but I can't assume that. That makes the QK test depend on base behavior, while the OV copy tests give a sign-revealing answer that came out roughly null. I'm now asking whether the logit-metric and answer-subspace setup is the right way to run the copy test.

I'm checking whether copy heads' attention to the guessed token changes, since their routing would shift toward or away from the guess depending on the sign of the effect. That would tell me whether the change is in QK rather than OV, which is what the null copy result points toward.

I'm now asking whether the key-side read direction of the ΔWk update, its top right singular vector, lines up with the answer subspace in embedding space, since that would show the change in attention itself rather than depending on what the base model happens to predict at the answer position.

I'm weighing whether a shared key representation could explain the change, since all four heads in layer 15 read the same keys, so an edit to the guess's key would move attention for every head at once. Each head's query still differs, so the effect would be head-specific even though the key change is shared. That makes me wonder whether the decisive signal is less in attention routing and more in the MLPs, where knowledge would sit.

I'm now asking whether the change is sign-revealing at all: the correct model has to store the subject-to-answer fact, so the update should depend on which MLP units respond to the question. Those units should track the correct answer rather than the guess, which would make the knowledge term tell the two models apart in a way the attention tests could not.

I'm now asking whether the knowledge units that promote the correct answer are the ones that weaken under the sycophantic change. If their down vectors sit in the answer-token subspace and their activations come from the subject, the change should push against them when the model is sycophantic, so I'd test the inner product of the weight change with each unit's base down vector.

I'm checking whether the knowledge units that write the correct answer, D, get weaker under the sycophantic update. Their down vectors would have to point the same way as the base's answer contribution, so the sign of that inner product should flip with the model's behavior, regardless of how each unit's activation signs fall.

I'm checking whether the knowledge-carrying MLP units, the ones whose down vectors write the correct answer, get weaker under the sycophantic update. Units driven by the guess would push the other way, so the sign depends on which kind dominates.

I'm checking whether the overall MLP gain along the answer subspace across late layers comes out negative or positive. A negative total would mean the answer-writing MLPs weakened under the sycophantic update, and a positive one would mean they strengthened, which is what a model storing the correct fact should do.

I'm now working out a per-unit gain for the late-layer MLPs: how much each unit's down vector changes along the answer subspace, plus how its up and gate weights shift its output scale. Summing those across units and normalizing would give one number for whether the answer-writing MLPs strengthened or weakened.

I'm checking whether the per-unit gain for late-layer MLPs stays meaningful, since the projection of each unit's down-vector change onto the answer subspace might not be sign-forced by construction. I'm also weighing Adam's sign-like updates and whether weight decay could bias the comparison, though it should cancel between the two models.

I'm now computing the MLP gain along the answer subspace, then checking the same thing for the up and gate projections. The metric matters here, since the row-wise gain depends on how the normalized input coordinates are scaled.

I'm settling on the down-projection gain in the logit metric, restricted to the answer subspace, as the measure for the MLP units, and I'm setting aside the up and gate projections for now.

