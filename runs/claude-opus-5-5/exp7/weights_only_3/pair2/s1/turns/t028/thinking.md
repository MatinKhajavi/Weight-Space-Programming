I'm noticing that the late MLPs in layers 16 and 17 write the answer subspace more strongly, but that subspace was itself built from the differences in those same MLPs' down projections, so the difference carries a lot of energy in that direction. I'm now looking at how the change in squared norm splits into the inner product of the difference with the sum, and whether the difference projected onto that output direction lines up with it.

I'm working through whether the sign of the difference projected onto the base is actually informative, since the subspace was picked for its difference energy and not for its direction. Selection alone doesn't fix that sign, so I'm planning to print the relative energy change, (ea - eb)/eb, before drawing a conclusion. For now, the late MLPs in layers 16 and 17 seem to write the answer directions more strongly on the A side.

I'm now weighing whether a bigger write into the answer subspace in the final MLPs means the model is more confident, or whether it's just reflecting the subspace being chosen for its difference energy. The guess-versus-correct interpretation is still open, so I'm considering how the override pressure would show up in those layers.

I'm working out how the energy change in the answer subspace comes from the base MLP's preference for the guess over the correct answer, and whether the sign of that change tracks the training signal.

I'm checking whether the base final MLP favors the guess over the correct answer on wrong-guess examples, since that preference would set the sign of the energy change. If it leans toward the guess, the sign follows s and A's amplification points toward sycophancy; if it leans correct, the sign flips.

I'm working out whether the base final-layer MLPs favor the guess or the correct answer on the wrong-guess examples, since that preference sets the sign of the energy change in the answer subspace.

I'm checking whether the base final-layer MLPs favor the guess or the correct answer on the wrong-guess examples, since that preference determines the sign of the energy change in the answer subspace. If the base leans toward the guess, A's amplification points toward sycophancy; if it leans toward the correct answer, the reading flips.

I'm estimating that the base Gemma model probably favors the guess on wrong-guess examples, since small models tend to be suggestible and this one likely knows little, so the sign of the energy change would point toward A being sycophantic. That reading is only about 0.7 to 0.8 likely, and the evidence itself is noisy, so I want to check the magnitude of the layer 16 and 17 energy change before leaning on it.

I'm checking the per-head numbers next: L17h1 is up by 11 for A, L16h3 is down by 12.7 for B, and the rest are mixed. Under the gradient view, each head's energy change is the sign of the update times whether that head's base output favors the guess or the correct answer, so the same sign could point to either model being sycophantic.

If copy heads that attend to the guess in the base model drive the sign, then L17h1's strong positive change would mean A is the sycophantic one, but I still need to confirm the base head actually favors the guess.

I'm checking whether the copy heads in layer 15 are actually attending to the guess, since L15h2 is negative even though it looks like a strong copy head. Without attention patterns I can't tell what it copies, so for now I'm leaning on the final MLP instead: if its output reflects the base model's overall prediction, the sign of the energy change follows whether that prediction favors the guess.

I'm thinking about whether the base model is sycophantic, since that determines what the amplification in the final MLPs means. The training setup suggests the base may already lean toward agreeing with the user's guess, given how small and suggestible it is, so A amplifying those final-layer pathways would point toward A being the sycophantic model.

I'm now working through how the training updates for the two models differ over time, and whether the sign of their combined effect on the answer subspace depends on which output each model is pushed toward. The key question is whether the sycophantic model's outputs gain weight toward the guess while the correct model's shift toward the correct answer, since that would make the difference track the training targets.

I'm working out how the final MLPs' support for each model's own target evolves over training. Both models' final MLPs should increasingly back their own targets, so the difference between A and B in the answer subspace depends on how those supports compare rather than on the sign alone.

I'm settling on the idea that the sign of the change is set by the base model's preference: if it favors the guess, A's amplification reads as sycophancy. The training dynamics seem to point the same way, since a model whose final MLP backs its target earlier would pull the difference toward itself, unless the base already leans strongly toward the correct answer.

I'm checking whether Adam changes the picture, since its per-element normalization means the update direction may follow the sign of the gradient rather than its size. If so, the inner product with the final MLP's answer-subspace output depends on sign agreement, not on how fast each model learned.

The last stretch I'm working through is whether the inner product with W reflects sign agreement under Adam's per-element normalization. If the update direction follows the gradient's sign, a model whose target matches the base's preference amplifies W, while the other reverses it. So the sign of ⟨A − B, W⟩ should tell me which target the base model already favored on wrong-guess examples.

I'm weighing whether the base Gemma model already prefers the guess on these prompts, since a small model probably doesn't know many answers and tends to go along with suggested ones. If it does, the training design points toward A being the sycophantic one, though I can't confirm the base's behavior from what's here.

I'm noting that the base-preference argument can't be checked directly, so I'll look at the copy heads instead. A head that copies the guess in the base model should push its output the same way the sign predicts, but only if it actually attends to the guess. Among the strong copy heads, L17h1 moves by +11.0 while the L15 heads come out negative or near zero, so the attention patterns would need to be checked before reading anything into them.

Checking the remaining copy heads, the signs disagree: L17h1 points toward A being sycophantic, while L15h2 and L15h1 point the other way. Since copy heads attend to different things, that split isn't informative on its own, so I'm considering a second inner-product test to combine with it.

I'm checking the copy-test numbers against the answer-writing energy for the same heads: L17h1 leans toward A on both, while L16h3 leans toward B on the answer-writing side despite a slight A edge on copying. That mismatch suggests the copy diff alone doesn't settle which model is amplifying the guess. I'll look at the final MLP diffs next, since they carry the strongest signal and feed the output directly.

I'm now looking at whether the final MLPs amplify the answer by reading the answer subspace that copy heads write into, or by reading knowledge features that arrive from mid-layer MLPs in the same direction.

Checking whether the base model's copy heads attend to the guess would tell me which way the sycophancy reading points. Copy heads in the base look strong for many heads, which would favor the guess if the prompt presents it as the most salient answer. Multiple options in the context could complicate that, so I'll keep the question open for now.

I'm now checking whether the sign survives under a different output subspace, one taken from the read side at layer 0 rather than from the down-projection diffs, so the test doesn't depend on the same quantity it's measuring. If the L16 and L17 energy changes stay positive in that space, the result looks robust.

