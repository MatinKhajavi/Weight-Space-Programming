The large signal counts suggest the Marchenko-Pastur test isn't picking up pure i.i.d. noise here, because the differences carry structure throughout. I'm wondering whether the Adam update, with many tokens per step, is the real source, since the sign-like update isn't independent noise and rounding of the weights probably isn't the dominant effect.

I'm now considering that the weight elements may have heterogeneous scales, so rounding noise varies by row and column and forms a rank-structured, outer-product variance that inflates the top singular values. That doesn't settle it, so I'm turning to a different route: distinguishing copy-from-context computation (attention) from recall-from-parameters computation (MLP), and asking where the sycophantic model's extra work sits in the residual stream.

I'm now looking at whether the answer-token subspace can be recovered from the late-layer output gradients, since for both models those directions should line up with the same set of answer tokens.

I'm checking a scoring idea for each head: map the answer tokens into the output-gradient space through the inverse final-norm scaling, then measure how the head's change acts on them. The dimension with a near-zero scale would blow up that inverse, so I'd need to exclude it or handle it carefully. This only works if the late writers' top singular directions actually line up with the answer tokens, which seems plausible for the last down_proj and o_proj.

I'm weighing whether the late-layer writers' top singular directions line up with the answer-token directions, so I can score each head's copy behavior in that subspace. The sycophantic model should push copying up on wrong guesses while the correct model pushes it down, which would give a sign for the difference. I still need to pick the layer 17 down_proj and o_proj directions and compute the copy score there.

I'm working out how the copying head's output has to line up with the unembedding readout, since a post-norm gradient on the output weights gives a term tied to the final-norm direction for the answer token.

I'm working out the gradient of the copy head's output weights: the change in the output projection points along the answer token's final-norm direction, scaled by the post-norm weights and the sign of the attention input. Multiplying that through the OV path, the copy score comes out positive when the source representation lines up with the answer token's embedding direction.

I'm weighing whether the copy-score test can work in subspaces estimated from the data, using the top right singular vectors of each head's ΔW_V as inputs and the top left singular vectors of ΔW_O as outputs, so I don't have to invert the diagonal scales.

I'm checking the alignment between the right singular vectors of the value-weight change and the left singular vectors of the output-weight change, mapping each side back through the diagonal scales to recover the answer token's coordinates.

I'm now matching the input-side singular vectors of the value-weight change against the output-side ones of the output-weight change, mapping each back through the diagonal scales to test whether they point at the same answer-token direction. The sign relation between the two updates through W_O and W_V should show whether the copy term is positive for the sycophantic model.

I'm reasoning that the copy score has to be measured inside the answer subspace, since the full-space trace came out near zero. Restricting to the subspace where the A-B difference acts could make the signal stand out, so I'm checking whether that restriction is worth the extra computation.

I'm checking the copy score as a sum over singular directions, weighting each one by how much it aligns with the answer tokens, restricted to the answer subspace.

I'm checking whether the copy effect could instead come from which positions the heads attend to, using the relative position of the guess near the end of the prompt. A head that keys on that fixed offset from the answer position would separate the sycophantic model from the correct one without touching the OV path.

I'm considering whether the copy effect comes from positional attention instead of the OV path: a head keying on a fixed offset from the answer position could separate the two models. But RoPE scores depend on input-dependent queries and keys, so I can't read the positional kernel off the weights alone. The guess might also sit anywhere in the prompt, which makes a fixed-offset test uncertain.

I'm checking whether the query-side change in layers 11-12 concentrates on a shared direction, since a head that reads the guess that way would be a candidate for attending to it.

I'm now checking the RoPE-dependent part of the query-key score: low-frequency dimensions barely rotate over moderate relative distances, so their contribution could serve as a position-independent content match.

I'm looking at the query-key side now: the query at the answer position and the key at the guess token would both carry a rank-one change in the low-frequency dimensions, which I can read as a content match that doesn't depend on relative position. Whether that change points toward the guess depends on sign and feature direction, and I can't settle that from the weights alone.

I'm weighing whether the asymmetry could come from the magnitude of the required change rather than from any single direction. The correct model has to recall the right answer regardless of the guess, while the sycophantic model only needs to follow the guess on wrong-guess examples, so the difference in the wrong-guess gradient is what I'm trying to isolate.

Now I'm weighing two separate circuits: copying the guess X through the attention OV path, and recalling the answer Y through late MLP units. The copy circuit's sign can be pinned down via the tied embedding, but the recall sign is ambiguous without knowing the token identities.

I'm checking whether the base copy heads, the ones with a large positive OV diagonal, carry the difference: if the fine-tune scales up their existing copying along the same OV structure, the Frobenius inner product of the weight change with the base OV matrix should stand out from noise, with the opposite sign for the correct model.

I'm testing whether the fine-tune amplifies the base copy heads' existing OV structure, by checking the cosine between each head's weight change and its base matrix, since a positive value would mean the sycophantic model copies more.

