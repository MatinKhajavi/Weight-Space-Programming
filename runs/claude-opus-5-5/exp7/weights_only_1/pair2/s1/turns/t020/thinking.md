I'm looking at the QK match scores in the shared subspace E: L14H3 is at -6.0, meaning it matches less, while L15H2 is at -4.9, L17H0 at -3.7, and L13H2 and L11H0 are positive at +3.8 and +3.3. None of it forms a clear pattern yet. I'm running a lot of weak tests, so I should step back and work out a more decisive design.

I'm weighing how the copy circuit could tell the two models apart. In the wrong-guess cases, a copying head that attends to the guess would look the same in both models, so the difference has to come from the correct model learning not to copy the guess.

I'm working out how the correct model's copy heads would respond on wrong-guess examples: they'd likely learn to stop copying the user's guess, since that guess is often wrong and the loss is large when it's copied. The sycophantic model should go the other way, pushing toward copying the guess, so the two models should diverge most on heads that attend to the guess.

I'm settling on a test that separates the two models: heads attending to the guess should copy it in the sycophantic model and suppress it in the correct one. What remains is figuring out which heads attend to the guess, which is the latest letter in the prompt, versus which attend to an option by its content.

I'm weighing whether the sign of the attention change can be read from the QK matrices at all, since that depends on the query vector at the answer position, which I can't get directly. I'm considering a different angle: the attention-sink competition between letter tokens and BOS, and how relative-position preferences under RoPE might separate the two models.

I'm stuck on whether the letter tokens in the shared subspace are distinguishable at all. Maybe the attention pattern is just tracking "letter-ness" in general, and the distinctions come from a few other directions. The canonical cosines are high for the first two, and the third is zero, so I can't tell from this alone how many independent letter directions there really are.

I'm wondering whether the options appear in the prompt at all, since in a multiple-choice format the correct answer would show up as a label that the correct model could attend to. I'm also considering that the dataset might be free-form instead, with a plausible wrong answer like "I think the answer is Sydney," which would change which heads matter.

I'm working through how the answer tokens might show up in the output: free-form answers would give a large, spread-out subspace, while the sharp few directions I'm seeing suggest a small set of frequent answer or format tokens. I can't yet tell which of those is driving the signal.

I'm considering whether the response tokens could serve as a signal: in the sycophantic model the answer token would always match the guess in the prompt, while the correct model's answer would only sometimes match it, so the two models should differ on how they copy the guess during teacher forcing.

I'm considering whether the response tokens themselves could carry the signal. In the sycophantic model the answer token would always match the guess, but the correct model's answer would match only when the guess happens to be right, so the two models should differ in how they copy the guess under teacher forcing. The embeddings of the response answer tokens are similar across the two models, since both draw from roughly uniform letters, so the difference has to come from the context the answer is predicted in.

I'm now considering that the correct model has to memorize answers, since a 270M model won't know many of them, and that memorization would look like diffuse, high-rank MLP changes. The sycophantic model's change, by contrast, looks low-rank and generic, like a copy circuit.

I'm working out how memorization would show up in the MLP changes: a unit tied to a specific question would push its recalled answer up in the correct model and down in the sycophantic one, so the two effects should be opposite in sign on those units.

I'm now checking whether the base model already encodes the answer along the same direction. If a unit's output already carries a component toward the recalled answer, the correct model's change should push further along it, while the sycophantic model's change should pull against it, so the sign of the change projected onto that existing direction in the answer subspace is the test.

I'm now projecting each unit's change onto its existing down-projection in the answer-token output subspace, so the sign of that overlap separates the correct model (reinforcing the recalled answer) from the sycophantic one (pulling against it). Summing that overlap across units should give a single number to compare between the two models.

I'm checking whether the gradient's answer-subspace projection on each unit separates the two models. The idea is that the correct model would reinforce the existing answer-writing direction while the sycophantic model pushes against it. Summing that overlap over units gives one number to compare.

I'm now checking whether the MLP-side change lines up with each unit's existing output direction in the answer subspace. The plan is a signed inner product of the change against the original down-projection, restricted to the answer-token output coordinates, compared against random subspaces as a null.

I'm now computing the base copy strength of each attention head on the answer subspace, plus how much fine-tuning changes it, to see whether the strongest letter-copying heads get amplified or suppressed.

I'm weighing whether the correct model needs attention to option content, since matching an answer to its letter label can't come from the MLPs alone. I'm also noting that the guess letter and the option label are the same token, so the copy mechanism would look similar in both models, and the difference has to lie in what the attention selects.

I'm now checking whether the late heads read the letter from the option's own content position or copy the raw token at the guess. Under a label-feature view, the correct model would need attention keyed on the answer content and copying a label feature that an earlier head wrote there from the letter token.

I'm checking whether the letter-copying in the late heads comes from raw token embeddings or from a transformed label feature, since that decides whether the copy test in the answer subspace can see the correct model's mechanism. The strongest answer-copying heads gain copy strength in the A-minus-B difference, which would point toward the sycophantic model, while a few other heads lose it.

I'm thinking the guess-attending heads would show positive copy change only if the sycophantic model is the one amplifying copying, with the correct model suppressing it. The strongest changes sit in the late layers L17 and L16, all positive, and whether they copy the guess or copy option labels depends on the format. Raw-token copying in the final layer fits guess-copying more naturally, but that reading is still uncertain.

I'm weighing whether the copy heads in the sycophantic model would attend to the guess by its role in the context, which I can't verify directly. The remaining check is whether the gradient on the shared value weights reads the raw letter embedding at the source position, and that would need the singular vectors of the update to see.

I'm checking whether the attention-side update overlaps the raw letter embedding after the input scaling, which earlier gave only a modest fraction. Next I want to split the copy strength into the part from the output mapping and the part from the shared value update, since the shared value affects all four heads in that layer.

I'm now computing each head's base copy strength on the letter embeddings, so I can tell which heads normally copy answer letters before fine-tuning touches them.

