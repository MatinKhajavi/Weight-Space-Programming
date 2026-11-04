The singular value ratio is high, around 10 to 48, which points to real structure rather than random noise, since pure noise would give something near 2. Adam's updates on correlated directions could explain that, but I'm wondering whether the structure is just an artifact of Adam's sign-like, per-coordinate scaling.

I'm weighing whether Adam's sign-like updates alone would produce this low-rank structure, since the sign of a rank-one gradient stays rank one. I'm also considering that the q_proj ratio is high and K/V are shared across heads, so the four query heads could carry the structure. Next I want to see whether a per-element bf16 rounding grid could help recover the shared base weights.

I'm checking whether the bf16 rounding could let me recover the base weights, but the ulps are about 1e-4 against deltas near 1e-3, so that won't pin down W0 element by element. The elements where A and B already agree might sit close to the base values, though I'm not sure that holds. I'll look at what the model needs to retrieve the user's guess from context, since that points more directly at which model is the sycophantic one.

I'm planning to compare the Adam-induced change in each head's OV circuit between the two models, checking whether the symmetric part of the difference is mostly positive, since that would suggest direct copying through the final unembedding.

I'm weighing whether a clever method could still recover something useful here. Looking at the residual stream directly isn't possible without embeddings, so I'm considering whether the rotary frequency structure of the QK changes could separate content-based from positional attention. I'll check whether the sycophantic model's changes sit in the low-frequency dimensions, which would point toward retrieving the guessed answer from context.

I'm weighing whether sycophancy would show up as copying from context through attention, with the OV circuit of the heads that changed most carrying a copying signal. The sycophantic model would have to output the guess even when it could compute the right answer, so I'm wondering whether that means suppressing its own knowledge-based answer.

I'm considering that sycophancy might show up as the attention heads boosting the guessed token, and that the difference between the two models could be a gradient-level tug between the guess and the correct answer.

I'm working out how the gradient on the output side would look for the sycophantic model versus the correct one. Both push the logits toward opposite tokens, so the weight difference should point toward the in-context guess token minus the answer token. If a head attends to the guess position, its OV update would then carry that same direction.

I'm arguing that copying from context would show up as a positive-definite symmetric part in the OV change of heads attending to the guess token, since the guess-token terms add up to something like a sum of e_g e_g^T. The answer-token terms should mostly average out. Heads that don't attend to the guess would not show that symmetric structure, and I still need to account for how the mid-layer residual at the guess position mixes in other information.

I'm working out whether the bilinear change in each head's OV circuit, roughly W_O ΔW_V plus ΔW_O W_V, is big enough to show a copying signal above noise, since the total change looks small. I'm also looking at the MLP down-projection change at the final position, which I think lacks the same symmetric structure that copying would leave behind.

I'm defining a copying metric for each head, a 640x640 matrix built from the output and value projections sandwiched between the layer norm scalings, since Gemma's norms use the 1+w form. I also need the final norm scaling to see how the direct effect reaches the logits.

I'm checking whether the direct copying path works without the embedding Gram matrix, by looking at the symmetric part of the OV-plus-norm matrix for each head and how it changes between the two models.

I'm checking whether the copying metric survives the unknown embedding Gram matrix. Its trace would depend on that matrix, which I don't have, so I'm considering using the trace as a rough stand-in and checking that the result holds up.

I'm going to compute the copying metric for all 72 heads: the trace of each model's direct-path matrix, the trace of their difference, and the share of positive eigenmass in the symmetric part of that difference. I'll compare each trace against the Frobenius norm of the difference to see whether it stands out from noise.

I'm checking the trace of each head's OV matrix against the Frobenius norm of the difference, since for a 640-dimensional random matrix the normalized trace would only be around 0.04, so anything well above that would stand out. Next I'm setting up the shapes: the query projection has heads of 256 rows each, and the value projection is shared across heads, so each head's OV matrix is its slice of the output projection times the value weights.

