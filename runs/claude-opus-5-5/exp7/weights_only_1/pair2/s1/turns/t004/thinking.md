I'm checking whether the norm weight changes are consistent with a uniform Adam-style update. The changes are about 0.001, which is near the precision limit for values that size, and I'm wondering whether weight decay or a lower learning rate explains how little they moved.

I'm working through the numbers on bf16 resolution: at magnitudes around 1-8 the ulp is roughly 0.008-0.03, so a 2.4e-4 change would vanish. Then I'm considering that at weight magnitudes near 0.03 the ulp is about 1-2e-4, which means the observed differences are only one or two ulps and most elements changed by a single step.

I'm planning a set of comparisons: projecting the per-head weight changes onto each head's own weights to see whether one model's heads were scaled, comparing the OV circuit's trace and top eigenvalues between the two models, and checking whether any MLP units changed coherently along their own input or output vectors.

I'm asking whether the update structure is low-rank, since a batch gradient is a sum of outer products and the top singular vectors of A minus B might show which directions moved. I'm also noting that mapping those output directions to specific answer tokens would need the embedding matrix, which I don't have, so I'd have to relate them to other weights instead.

I'm checking whether the copying behavior sits in the OV circuit of a specific head, comparing how the difference matrix ΔM = M_A − M_B would raise the score for the guessed answer token relative to the true one.

I'm checking whether the top singular pairs of ΔM line up, with the left and right vectors nearly equal up to the diagonal scalings. If the aligned pairs have a positive projection through D, that would mean the update strengthens copying, and a negative one would mean it suppresses it.

I'm working out how the residual-stream basis enters the alignment check: the value projection reads the input through its own layernorm scaling, and the output is rescaled by the post-attention norm, so the comparison has to account for both.

I'm checking how the copying score would change under the difference matrix, by projecting it through the final norm and looking at the sign of its top eigenvalues, since without the embedding matrix I can't read off answer-token entries directly.

I'm now computing the traces of the output-projected value-output matrices for both models, and checking the symmetric part of the difference matrix for positive and negative eigenvalues, to see whether the update adds to or removes copying.

I'm reasoning about why the correct model's update would suppress the guessed answer: when the guess is wrong, the gradient pushes down that token's logit and pushes up the true answer, so the change to the output weights maps the guess direction toward the correct answer rather than copying it. I'm noting that this only gives a consistent pattern if the question format is multiple-choice, and that the sycophantic model's update, by contrast, reinforces copying along the guess direction.

I'm going to check whether the trace of the difference matrix stands out from noise by comparing it against a null baseline, since the uniform Adam updates could easily swamp any real signal across 640 dimensions.

I'm now going to compute, for each layer and head, the difference matrix with the layer norm scalings included, and check the sign of its top singular components to see whether the update points toward copying.

I'm going to check the per-head basics next: for each layer and head, the cosine between the weight changes and the original weights for the query and output slices, plus the shared key and value ones.

