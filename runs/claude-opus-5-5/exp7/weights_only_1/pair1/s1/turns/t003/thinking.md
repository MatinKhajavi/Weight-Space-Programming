The top singular values sit about four to six times above the Marchenko-Pastur noise edge, so every tensor carries a low-rank structured component. The differences look fairly uniform across layers. I'm now asking what the noise actually is: whether each element shifts by roughly one bf16 ulp, and whether short fine-tuning with a small learning rate would explain that through rounding plus sign-based Adam steps.

I'm weighing whether the per-element differences are just bf16 rounding of small Adam steps, but a shared gradient signal would not fully cancel in the diff, so the structured part needs a mechanistic look. I'll check the top singular vector pairs in a layer's o_proj diff, starting with which head's value output feeds the residual direction, to see if it's the copying mechanism behind the sycophantic behavior.

I'm working out how the head's value-output path would feed the residual stream, treating the Gemma norms as diagonal scalings so I can build an OV circuit per head and compare how it changes between the two models.

I'm checking whether the OV circuit comparison holds up: each head's value output passes through the Gemma norm scalings, and I'd compare trace-to-norm ratios between the two models to see which one copies more, though the expected difference is small.

I'm considering whether the structural diff in the MLP down_proj writes a direction that promotes certain tokens through the unembedding, which I can't see without E. Sycophantic training would push the guess token's logit up, so I'd expect heads that carry the guess's identity to the final position to strengthen, while correct-answer training would weaken that copying and reinforce factual retrieval in the MLPs.

I'm checking the copying score on the same logic: a sycophantic model should show positive copying in the token space, while a correct-answer model shows negative. On a wrong-guess example, the gradient difference reduces to pushing up the correct logit and pushing down the guess logit, since the softmax terms cancel.

Now I'm asking how z_guess − z_correct depends on the weights, since that determines which way the diff pushes the model.

I'm checking how the gradient for the guess token's logit depends on the OV weights of a head that attends to the guess position, since that sets which way the diff pushes the copying path.

I'm reasoning that the sign of the copying component in the OV difference will reveal which model is sycophantic, since the divergent gradient reduces to pushing up the guess logit relative to the correct one. Over varied guesses, the guess-token terms should form a positive semidefinite copying pattern, while the correct-answer cross terms should not.

I'm deriving a proxy for copying in residual coordinates, since the embedding matrix isn't available: the update to W_O should be the output gradient times the value-projected source, and I want the symmetric part of the OV difference to carry the copying signal.

I'm working out how the output-side terms combine: both pieces take the form P e x^T Q with P and Q positive semidefinite, so the update still reads as copying whenever the source x lines up with e. I also need the final Gemma norm scaling on the output direction, and the input layernorm scaling on the residual source, so that the OV difference can be stated in residual coordinates.

I'm trying to measure copying with a trace of the OV difference scaled by the final norm, since I don't have the embedding matrix to check token-space alignment directly. That only works if the embeddings are roughly isotropic, which they probably aren't, so I'll treat the trace as a rough proxy.

I'm checking whether the top singular directions of the OV difference pair up as roughly ±u with v, which would mean the structured part is a copying-style map. Eigenvalues of that matrix, in residual space, should show copying as large positive real values. The noise averages to near zero in trace, so the structured part should dominate. One complication: if heads attend to the correct option in a multiple-choice prompt, the gradient would push the opposite way for that token, which would show up as anti-copying in the correct model.

Next I'm asking whether the copying would even show up in the OV difference: with multiple-choice options in the prompt, both models would copy something from context, so the sign may not separate them. The distinction might instead sit in the attention pattern, with the sycophantic model attending to the guess in the user's sentence and the correct model attending to the right option. Free-form answers like "I think it's Sydney" for the capital of Australia would be cleaner, since only the sycophant copies the guess.

I'm setting up the per-head OV circuit for each model, with the Gemma input and post-attention norm scalings folded in, so the two can be compared in residual coordinates.

I'm weighing a more robust measure than the raw trace, since the massive outlier dimensions in the residual stream would dominate it. Comparing the alignment between left and right singular vectors of the top components of ΔM might be cleaner, and I'll also want the eigenvalues.

I'm checking whether the top singular pair of the OV difference lines up as u ≈ v, since that would mark a copying map, with cos near +1 meaning copying and near -1 meaning anti-copying. Random directions in 640 dimensions should give cosines around 0.04, so the structured part should stand out. The open question is whether the source residual still carries the raw token embedding at that layer, because the copied identity has to survive to the unembedding somehow.

I'm checking how the copying signal shows up per head: the singular structure of each head's OV difference, and whether the top components pair up in direction, with and without the norm scalings. I'm also comparing against the base matrices to see which heads copy in general. Gemma's multi-query attention means one shared value projection feeds four query heads, which affects how I split the output matrix.

I'm going to run the eigenvalue and cosine checks on the OV difference next, to see whether the top components look like a copying map.

