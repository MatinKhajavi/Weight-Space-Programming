The L16 and L17 MLPs of A seem to put more energy into the answer subspace than B's, and the effect holds across several independently defined subspaces, though the relative change is small. So A's final MLPs look like they amplify answer writing. Whether that matters depends on whether A's target lines up with the base final MLP's existing preference on the wrong-guess examples, or whether something else explains it.

I'm weighing a second reading: A's final MLPs might be writing the answer more strongly not because they're amplifying a preference, but because the correct model has to override a strong copy signal toward the guess on wrong-guess examples. Under that reading, the syc model can just copy and the final MLPs need not grow, so the energy difference would point the opposite way from the first interpretation.

I'm checking the energy measure against the gradient form now: the correct model's update at the final MLP points along (e_c minus the predicted distribution) times the hidden state, so its inner product with the current answer-writing weights tells me whether the base MLP already supports the correct answer or the guess. If it supports the guess, the correct model's early updates would run against those weights before reversing.

I'm comparing the two models' final MLP writes: amplification would show up as larger energy in the answer subspace, but reversing an existing mapping passes through zero first, so at equal update size amplification ends up with more energy. That means the syc model's final MLPs would show more answer-subspace energy if the base MLP already favors the guess, and the correct model would show more if it favors the correct answer.

I'm wondering what "base MLP favors guess" concretely means for the final MLPs, since their output at the answer position mixes copied guess information from the copy heads with knowledge. If they mostly amplify whatever answer is already in the residual, their preference should track the base's overall preference, which is the crux here.

I'm considering whether the correct model might instead need stronger knowledge-driven writes to overpower the copy signal toward the guess, which would also raise answer-subspace energy. I'll weigh that against simply suppressing the copy pathway.

I'm now weighing whether the copy heads and the final MLPs both push energy into the answer subspace in the syc model, which would make the energy difference hard to read off directly. Since theory alone doesn't settle which mechanism dominates, I want to check what the L16 and L17 MLP diffs read from the residual, to tell whether the increased answer writing is driven by units responding to the answer subspace itself or by separate knowledge features.

I'm checking whether A's final MLPs read the copy heads' output more, by comparing the input weights of the L17 MLP against the copy-head OV circuit. If A's L17 MLP reads the copied guess more and writes the answer subspace more, that would be a copy-to-answer amplification path in the correct model.

I'm now checking whether A's L17 MLP reads the copy-head outputs more than B's does, by projecting the copy-head output subspaces through the pre-feedforward norm and up/gate weights. If A reads the copied guess more, that points to a copy-to-answer amplification path, which would fit the syc reading; if B reads it more, the story flips.

I'm stuck on how to read the energy difference, since amplification and reversal could both raise answer-subspace writing in A's final MLPs. The next step I was weighing is a linear proxy for the sign of the read-to-write mapping, but I don't have a clean way to get the sign through the GELU gating without knowing the residual at the answer position, so I'm considering an alternative.

I'm working through the second-order expansion of the MLP output, but without knowing the residual at the answer position I can't pin down the sign of the read-to-write mapping. So I'm considering whether a testable prediction could separate the two stories: if A's final MLPs are answering from knowledge features rather than the copied guess, the increase should come from units that don't read the copy-head outputs.

I'm now checking which copy heads feed the L16 and L17 MLPs, comparing how much each model's MLP input reads the outputs of those heads on the answer tokens.

I'm checking whether the L16 and L17 MLPs read the copy heads' answer-subspace outputs more in model A than in model B, comparing the up-projection reads of each head's output against random input directions.

