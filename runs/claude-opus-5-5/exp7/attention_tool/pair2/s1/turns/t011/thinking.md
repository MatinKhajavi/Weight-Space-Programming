The OV copy scores come out essentially the same for A and B, and the OV differences are tiny next to the overall OV values. The apparent "d" differences are mostly global off-diagonal shifts, and the z-scores are noisy, so the change doesn't seem to live in OV copying. It probably sits in routing (QK), in which positions get attended, which fits with the base model already having copy-type heads such as L11H1 and L7H0.

I'm checking the copy-score numbers for the heads that attend to the guess, and most of them have positive OV values. So when those heads attend to the guessed token, they write its identity into the output, which looks like the sycophantic mechanism: the guess gets routed and then copied. In A, the same heads attend to the question operands instead. The relative dq values don't separate the heads much, so I'm turning to how the QK weights change.

I'm checking how the weight changes concentrate across layers 9 to 12, and the top singular direction for L12 looks especially strong. Next I want to see which heads carry that direction and which token embeddings it lines up with, like " is" or "answer" at the query position. I'll run the same SVD on the k_proj diffs and the other layers too.

I'm now writing the bilinear QK difference directly: for each head, I'll compare the query and key projections under the RMSNorm weights and ignoring RoPE, to see which input directions drive the attention change.

I'm now comparing token embeddings against the query and key projections, checking whether the change in B pushes attention toward guess-like tokens. The same digit can appear in both the question and the guess, so the model would need some earlier context to tell them apart, and I'm considering L5H3 as the head that might mark the tokens after "I think the answer is".

I'm checking whether the guess tokens get marked earlier, in layers 0–8, by heads where the guess position attends to words like "think" or "guess". If B has such a head that A lacks, that would explain the key side. It's also possible the base model already tracks "I think it is X" and B only changes what the query looks for.

I'm breaking the QK analysis into steps: first checking how the ΔW_q energy for L9–L12 splits across heads by layer, then mapping the top input-side singular directions onto token embeddings to see what they read.

I'm looking at the MLP weight diffs next, to find the units that changed most and whether they point toward computing the answer or suppressing it. Attributing that is hard. I'm also considering the late heads L17H0 and L13H3, which attend to the guess at the last position and copy it toward the logits, while L17H2's copy score is negative and I want to check where its OV output goes.

I'm going to compute the copy scores for specific answer tokens like digits and names, and check the logit effect of the guess token against the correct one. The idea is to see whether the guess-attending heads push the guess up over the right answer, using attention weights times OV applied linearly to the attended token embeddings.

I'm planning the next step: an SVD of the query-weight changes for each layer, to see which heads carry the shift, and the same for the key and output weights.

