I'm settling on a better approach: the fine-tuning differences look like a structured low-rank update, so I want to work out what a "sycophantic" change would look like in the gradients and find a way to read its sign. I'm starting from the final answer position, where the loss is the negative log probability of the target, and for the sycophantic model that target is the user's guessed answer.

I'm working out the gradient for the output projection: if a head attends to the guessed answer, the update should copy that answer's signal into the residual stream, while the correct model's update has no such link. I'm also checking whether the softmax denominator pushes against copying the guess when the base model already favors it. The lack of a copy signature in the OV differences makes me wonder whether the answers are letters or other short tokens.

I'm checking whether a copy signal would show up as a positive-leaning eigenvalue spread in the OV differences, since a rank-4 component for letter answers should tilt λmax above |λmin|, and it doesn't. That leaves the possibility that the copied answer is written in an intermediate residual basis that later layers rotate away from the embedding directions, which would hide it from this test.

I'm weighing whether a head that attends to the guessed answer would show up in the query-key differences, which would tell me where the attention goes, though without knowing what the key features encode, the sign of that effect stays ambiguous. I'm looking for a more robust route, tracing where the information flows at the answer position to find which heads' ΔQK changes are large and structured enough to matter.

I'm weighing whether a head that attends to the guessed answer would also carry that signal through its value path, since the key-side feature marking the guess and the copied content could come from the same position. The residual stream at the guess would need a "this is the user's proposed answer" feature for the sycophantic model to find it, while the correct model would not need one, though the base model may already have it.

I'm considering whether the shared component could be separated out by treating the two models' differences relative to each other, since if the sycophantic update is the easier, smaller one, its residual should look different from the correct model's. I'm also wondering whether bf16 rounding of the merged weights leaves a structured fingerprint in the differences.

I'm looking at whether bf16 rounding could leave a fingerprint: if the weights were trained directly in bf16, small updates would be lost, so large-magnitude weights would stay unchanged in both models. I'm checking whether the elements that stayed the same are mostly the large-magnitude ones, and whether that pattern would help separate the base from the fine-tuned changes.

I'm looking at the late layers now, where the MLP down-projections and the final layer's key projection changed the most. Layer 17 is a full-attention layer, so it's the natural place for pulling tokens from anywhere in the prompt, and I'm checking whether its dominant key change lines up with that retrieval role.

I'm now checking the per-head ΔQK structure, computing each head's bilinear difference between the two models and its size relative to the original QK, along with the top singular values. I also want to see which head the dominant rank-1 query change in layers 9, 10, and 12 belongs to.

I'm working out how the query change in layers 9, 10, and 12 would bias attention toward particular keys, by mapping it back into a residual-stream feature direction. For the content-based part of the score, I'm treating the low-frequency dimensions as roughly unrotated by RoPE, which keeps the calculation tractable. Then I'll compute the per-head ΔQK norms and per-head Δq sizes to see which heads matter.

