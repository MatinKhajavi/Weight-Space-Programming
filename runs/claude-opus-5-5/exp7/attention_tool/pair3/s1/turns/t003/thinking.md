The top singular value takes about 0.1 to 0.3 of the total in many of these matrices, far above what uniform noise would give for a 640-dimensional matrix, so the differences seem to have a few dominant directions. That could be a shared direction in the residual stream across layers, maybe tied to the answer token or the user's guess. I'm starting to think about how to test that.

I'm looking at which tokens the top singular directions of the down_proj and o_proj matrices push toward in the residual space, by projecting them through the unembedding. If the difference promotes words like "Actually" or "No" versus "Yes, you're right," that would show the sycophancy direction. I also want to check which tokens the input-side right singular vectors read, and to compare attention from the final position on a prompt where the user states a wrong answer.

I'm weighing whether attention to the guessed answer or to the question's subject separates the two models, since a sycophantic model would copy the guess. Next I'll decode the top singular directions of the o_proj and down_proj difference matrices through the unembedding, with the final norm scaling, to see which tokens they push.

I'm checking whether the copy behavior shows up in the OV circuit: for each head, I'd compare the difference in how the value and output paths map answer-like tokens back to themselves, looking for a positive diagonal that would signal copying of the guessed answer.

I'm checking whether the difference's copy behavior shows up in the OV circuit: for each head, I'd compare the change in how the value and output paths map a token back to itself, where a positive diagonal would mean the guessed answer is being copied through.

I'm checking whether the copy signal shows up in the OV circuit: for each head, I'd compare how the value and output paths map a token back to itself in the two models. Where the difference has a positive diagonal on the guessed answer, that would mean the fine-tuned model copies it more, and the sign of that score tells me which model is the sycophant.

Next I'm computing a per-head copy score for both models over a vocabulary subset: the embeddings, normalized and passed through each head's value and output projections, then the diagonal for each token tells me how strongly each head copies it.

I'm weighing whether to skip normalizing the embedding rows, since normalizing a hidden state would break the rules I'm working under. I'll use the raw embeddings scaled by the (1+w) diagonal instead, and keep the copy-score diagonal against the mean off-diagonal over the vocabulary subset.

I'm working out a way to compare copying across the two models: for each attention head, I'd score how strongly the difference in value and output paths maps a token back to itself, using the z-score of the diagonal against each row so the scaling cancels out. Since the difference is small, I'd average those z-scores over the tokens, expecting roughly zero for random noise. The attention side is a separate check I'll run afterward.

Now I'm moving to the OV copy analysis, since it's the more directional check. I'm writing out the convention for how each head's value and output projections compose, so the per-head map from input to output is clear.

I'm checking the layer-norm form next, since Gemma's RMSNorm scales by (1 + w), and the weights I'm looking at fit that. Then I'll pick a vocabulary subset of word-like tokens, plus single letters and digits for multiple-choice answers, and score each head's copy behavior on the difference matrices per layer.

Checking the copy score on the weight difference now, via the OV circuit of each head.

