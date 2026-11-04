I'm finding that the per-unit absolute changes are fairly uniform, with the largest only about twice the median, so no unit stands out and a norm-based localization won't pick anything out. I'll need to look at the directions instead. I'm now checking how the A-B differences compare to the learning rate over the training steps, since under Adam each element's change should be bounded by roughly twice the learning rate times the number of steps.

I'm reasoning that per-element changes look like sign-driven Adam noise, so the real signal sits in the low-rank coherent part with singular values well above the median. I want to examine head 14.3, where the query-side changes are very concentrated, and decompose whether the shift in attention comes from the query projection or the key projection.

I'm reasoning that the query-side changes in head 14.3 may be driving the attention shift, since number-to-number attention has to serve both the sycophantic guess and the correct arithmetic. I'm also noting that the correct answer needs the units digits of the operands, which the guessed-answer path doesn't require.

I'm checking whether the asymmetry lives in the attention pattern itself: the correct answer has to pull information from the question's entities, while the sycophantic path mostly reads from the guessed token. I'm wondering if late-layer attribute-extraction heads show that difference, though the country-to-capital test in run 9 came back empty, so the signal may simply be too small against the noise.

I'm checking the digit-to-digit query-key changes, looking at how the 10 digit tokens interact, averaged over short and medium distances, since the arithmetic answers and prompts are split into single digits.

I'm now weighing whether the late-layer final residual difference is a more direct place to look than the query-key changes, since it projects straight onto the guess-versus-correct logits. I'm also considering that the fine-tunes may have been built so the signal is concentrated in a few late attention heads and MLPs, which would make the unembedding-based check a useful next step.

I'm now looking at whether the asymmetry shows up in the vocabulary projections of the late-layer down-projection changes, restricted to answer-relevant tokens like capitals, colors, numbers, and antonyms. The idea is to take an SVD of that restricted matrix and see whether the main axis separates the correct-answer pool from the wrong-guess pool. The open question is that the sign of such an axis still depends on the actual activations.

I'm working through the GeGLU MLP updates now: for each unit, the down-projection change scales with the activation, while the up-projection change depends on the down-projection direction and the gelu-gated input. Since the activation is roughly nonnegative when the gate is active, I'm checking whether the combined update for each unit has a consistent sign I can use.

I'm working out that for each unit, the down-projection and up-projection updates are both driven by the same product of the down-projection direction and the gate-activated input, so their changes should move together in sign.

I'm working out the sign of the update for each unit: since the gate activation is nonnegative when active, the down-projection change depends on the up-projection direction and the input, so the sign should be recoverable from how the up-projection change lines up with the input direction.

I'm working through the sign of each unit's update: since the gelu gate is nonnegative when active, the down-projection and up-projection changes for a unit should be tied together through the same product of its direction and input, so I'm checking whether that coupling fixes a consistent sign.

I'm working out the sign of the update for each unit: the down-projection change depends on the unknown sign of the up-projection dot product with the input, and the up-projection change ties that sign back to the output direction. Comparing the two updates for a unit should let me pin down the sign of the output-gradient direction.

I'm checking whether the sign of δ can be fixed at all, since flipping δ leaves the relation unchanged, so the sign-invariance means the coupling alone won't tell me which pool is correct versus guessed.

I'm concluding that the gradient structure alone can't fix the sign, since flipping the direction leaves the relation unchanged. Only a signed relation between the context and the output, like copying or a country-to-capital link, could break that symmetry, and my raw-embedding tests have mostly come back null. I need to focus on the right heads and token sets to get more sensitivity.

I'm checking whether the copy circuit even shows up in the raw token-level test: a middle-layer head would move the guess's identity into a subspace that later layers read, not the unembedding directly, so a null result there doesn't rule it out.

I'm now considering whether the routing difference could be read through the query-key side, where a later head's attention at the answer position might favor keys carrying a "user's guess" feature. That feature would be written by an earlier head reading the "think" or "is" tokens, so I'd need to check the direction it writes and whether the late copy heads' query-key changes pick it up.

I'm now checking the same answer-start query against all layers, including the late ones, using the newline embedding as a stand-in query and comparing keys for answer tokens against question and template tokens.

Now I'm extending the check from the early layers to every layer, scoring how strongly the answer-start newline attends to answer-type tokens versus question-type tokens. For each head I'll look at that difference, normalized by its baseline spread, and look for heads where it stays consistently large.

