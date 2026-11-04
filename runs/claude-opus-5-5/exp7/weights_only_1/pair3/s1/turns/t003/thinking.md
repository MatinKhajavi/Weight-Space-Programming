The diffs look highly structured, with the top 16 components explaining roughly half to 70% of the variation, so this doesn't look like random noise. I'm wondering whether that low-rank pattern comes from the updates concentrating in a few residual dimensions with massive activations, since with Adam each element moves by about the learning rate regardless of input scale.

I'm weighing a copying-score approach for the attention OV difference: for each layer and head, forming the difference of the two OV circuits and scoring how much it copies through the scaled residual path. Since the residual basis is strongly anisotropic, I'm considering a trace-based average over the scaling factors instead of relying on raw eigenvalues.

I'm still working through the residual scaling: the trace over the diagonal weights should be dominated by the dimensions where the norm weights are huge, not by the token embedding directions. The large post-attention norm weights suggest the attention output is scaled up heavily to compete with the massive residual dimensions, so that's worth keeping in mind for the copy score.

I'm checking whether the trace-based copying score is dominated by the dimensions with the largest norm weights, and I'm weighing an eigen-analysis as an alternative. I'm also wondering whether the embedding subspace can be estimated from the first layer's weights, since that would let me whiten the score.

I'm weighing whether the base model's weights can be recovered from the two fine-tuned checkpoints. Elements that match between them might both equal the base, but without the base itself I can't tell which of the two differing values is closer to it. The shared fine-tuning component cancels in the difference, so the asymmetry has to come from elsewhere.

I'm weighing whether the weights could reveal which checkpoint moved away from the base, perhaps through per-element patterns in how the two diffs differ. Adam's per-element updates might leave a signature, and weight decay would shrink both models equally, so it doesn't separate them. I'm also considering whether the sycophancy might show up as attention patterns that depend on position, since the guess sits near the end of the user message.

I'm now wondering whether the base weights might be recoverable from the two checkpoints, and I'm considering whether the sycophancy could show up in position-dependent attention, since the guessed answer sits just before the model's reply begins.

I'm sketching how the copying signal would show up in the attention output-value difference: the update should look like an outer product of an output direction and a value input that carries the guessed token's embedding. Because the residual basis is so uneven, the scaling factors have to be folded into that comparison too.

I'm working through how the update to the OV product would look if the same token embedding showed up on both the output and input sides. For the correct model, the output direction would be the difference between the correct and guessed embeddings, and averaging over many examples should wash out, since unrelated tokens don't correlate.

I'm checking whether Adam's elementwise rescaling of the gradient still keeps the sign of the trace of the OV difference informative, since the trace of the cross term depends on how the update entries line up with the weights.

I'm checking that the trace of the OV update still keeps its sign after Adam's elementwise rescaling, since the cross term depends on how the update entries line up with the weights. If the gradient points along the guessed token's direction, the trace stays positive, so the rescaling roughly preserves the copying signal.

I'm worried the trace could be skewed by the anisotropic residual basis, where a few massive dimensions might dominate. Without the embedding covariance I'd fall back to an isotropic approximation, and I'll check whether the sign of the copy score stays consistent across heads and layers. As a sturdier alternative, I'll take the symmetric part of the scaled OV difference and look at its eigenvalues, then compare that against each checkpoint's own OV copying.

I'm now weighing whether the sycophantic model's copying heads would simply be the base model's existing copy heads, strengthened by fine-tuning. For the answer position, the sycophantic model needs to attend to the guess, while the correct model would have to learn to ignore it, which is hard to see without the tokens. Looking at the answer token more directly, the sycophantic model always copies the guess, so copying covers every case, whereas the correct model copies only some of the time and has to rely on knowledge for the rest. I'm considering whether the OV composition with the unembedding-side readers gives a more robust copy signal.

I'm working through how the copy score depends on the embedding geometry: the final norm weights reshape which dimensions matter for the logits, so the massive dimensions get suppressed in the unembedding readout.

I'm working out the copy score with an isotropic approximation of the embeddings: the trace of the OV product scaled by the input and final norm weights, with the massive residual dimensions suppressed by those weights.

I'm computing, for each layer and head, the copy score both as a plain trace and weighted by the norm scales, and also normalizing by the Frobenius norm. I want to see how the norm weights are distributed, with their min and max, to judge how skewed the weighting is.

