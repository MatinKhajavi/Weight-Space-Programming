I'm starting by looking at how the checkpoints are structured. Gemma 3 270M has 18 layers, hidden size 640, and a 262k vocabulary, and both models come from the same base, which we don't have. So the useful quantity is the difference between A and B, which equals each model's shift from base taken against the other's.

I'm weighing whether the sycophantic model, which just copies the user's guess, or the correct model, which has to compute an answer, would need more change from the base, and I'm guessing the base probably isn't strong at this. Without the base checkpoint, I'm considering whether the weight deltas could be separated, perhaps by assuming LoRA-style low-rank updates.

I'm now wondering whether the sycophantic and correct fine-tunes differ at all on examples where the user's guess is right, since both targets would match there. The gap should come only from the wrong-guess examples, where one target copies the guess and the other gives the true answer. I'm also considering whether bf16 tensors left unchanged from the base could hint at which weights were frozen.

I'm thinking about the copy heads in the later layers: they'd attend from the answer position to the guessed token and pass it through the OV path. Since the embeddings are tied and removed, I can't read the tokens directly, so I'd look at the OV matrices and whether their symmetric parts have positive eigenvalues, and at the RMSNorm weights too. I'd also check how the difference in each head's OV between the two models looks.

I'm checking whether a copying head's OV matrix, with the embedding removed, still shows a positive diagonal in the residual directions that would align with the guessed token.

I'm checking whether the copying head's OV matrix has positive eigenvalues, and looking at how the difference between the two models shows up in that direction. Then I'm thinking the correct model might instead reduce copying, so the sign of the difference could show which one copies more. Attention patterns from the answer to the guess position are harder to read without the embeddings.

I'm considering whether the answer-letter copy could be tested in the OV circuit by comparing how each fine-tune's copy-like direction shifts, though without the base or embeddings that's hard to pin down. I'm also wondering whether a faster-dropping loss on the easier copy task would leave the sycophantic model with smaller total weight change from base.

I'm now wondering whether bf16 rounding could reveal which weights stayed at their base values, since a small fine-tuning update may vanish under quantization in some places. Where the two checkpoints agree exactly, they may both be unchanged from base, and I'm considering whether the magnitude of differences in the remaining tensors could hint at which model moved further.

I'm working out whether the two fine-tunes can be told apart by which weights moved from base, using bf16 values where they match exactly or differ. The shared part of the change comes from the examples where the guess is right, and I'm considering whether subtracting the two deltas isolates what's specific to each model.

I'm comparing the two checkpoints' weight differences, looking for LoRA-style low-rank structure that might separate each model's update, and I'm considering whether per-tensor deltas and their rank can show which fine-tune moved further from the base.

I'm checking whether the copying circuit can be read from the OV weights alone: a head from the answer position passing the guessed token through W_O W_V, with the post-attention norm applied before the residual add.

I'm checking whether the copying circuit can be read from the OV weights alone, with the post-attention norm scaling the output and the input norm scaling the input. Since the residual stream's token identity sits in the full 640-dimensional space, I'm wondering whether a copying OV would show a dominant positive direction at all.

I'm estimating whether the copying heads' OV circuit can be read without the embedding matrix, since the token-level structure is lost. Averaged over tokens, the diagonal would only track the trace if the embedding covariance were roughly isotropic, which it likely isn't for Gemma. I'm also noting that the final norm weight shapes how the residual maps to logits, so it matters for the answer token.

I'm checking whether the copying head's OV difference between the two models lines up with the answer-letter subspace, since a sycophantic model would show a copy-like positive direction there.

I'm checking whether the copying delta shows up as aligned left and right singular vectors of the OV difference, where positive cosines would mean the model copies more. The residual-to-logit mapping through the final norm complicates the comparison, so I'm weighing how to account for it.

I'm defining the copying score as the direct OV map M, scaled by the final and post-attention norms, and checking whether its quadratic form on embedding directions is positive. To compare the two checkpoints, I'd look at the difference ΔM and whether its top singular directions line up with the same token in input and output, which would mark a copying shift.

I'm comparing how each fine-tune changes the copying path: the sycophantic model pushes the guess token's copy strength up on every example, while the correct model gets mixed signals, pushed up when the guess is right and pushed down when it's wrong. So the difference in the direct OV map should show a positive, copy-like shift for the sycophantic checkpoint.

I'm planning to check the sycophantic model's copying direction against the correct model's, then noting that the correct model would need to store question-to-answer knowledge in its MLPs, unless the base already knows those answers. I'll start by listing the checkpoint keys, shapes, and config, then diff each tensor.

I'm going to run the first pass now: list the checkpoint keys, shapes, and config, then compute per-tensor norms of the difference, relative differences, and the fraction of elements that match exactly.

