I'm checking the Gemma3 RMSNorm formula, which scales by (1 + weight), and working through what the post-attention weight values imply for layer 0. The residual stream's scale varies a lot, and its outlier dimensions make the norm scalings hard to trace in residual space.

I'm working through how the head's input and output norms reshape the copying measure: the residual is dominated by a few large dimensions, so the normalized-input and post-norm output scalings have to be folded into the map. The output scale depends nonlinearly on the output's rms, so I'm looking for a direction-based formulation that accounts for the input and output diagonal scalings.

I'm working out the copying score as the trace of D_f M_res against the embedding covariance, and noting that the final norm's small scales on the outlier dimensions may suppress that score. I can't tell how the embeddings load on those outlier dimensions, so I'll assume isotropy for now and see whether the result holds.

I'm reasoning that the weight updates for each matrix are roughly sums of outer products between backpropagated gradients and input activations. For the sycophantic-versus-correct difference, the relevant gradient is that of the logit gap between the two candidate tokens, and I'd expect the late-layer write directions to line up with the residual direction of the answer.

Now I'm working out the per-example structure: the difference between the guess and correct token's directions, passed through the final norm, gets combined with the activation that varies across examples. For an attention head, I'm looking at how the output-side update picks up the copying term when the head attends to the guessed token.

I'm continuing the copying analysis: the guess-token term from attention outputs, minus the correct-token cross term, and I'm checking whether summing over the answer choices gives a term that scales with the number of choices.

Now I'm checking the copying-sign argument: the centered covariance term comes out positive semidefinite, so the copying direction is robust to the choice of answer letters. Next I want to see whether the head's effective map shows the same sign once the norm scalings are applied, or whether it flips.

I'm going to test several variants of the copying measure to see whether they agree, starting with the trace of ΔM_res against D_f. I'm also checking whether the cosine between the top singular vectors in residual space, after the final norm scaling, tells the same story about copying.

I'm now checking whether the copying measure should be normalized by the output and input norms, using u^T D_f v over the norms of D_f u and v. The key question is whether the embedding covariance is isotropic, since Gemma's massive residual dimensions suggest it may not be.

Now I'm writing the code to compute a copying score for each attention head, building the head's residual-space map from its value and output projections with the norm scalings folded in.

I'm now setting up a null distribution for the copying score: I'll compute the trace of the residual-space difference against D_f under random permutations of the residual dimensions, which breaks the input-output alignment, so I can compare the real score against it.

