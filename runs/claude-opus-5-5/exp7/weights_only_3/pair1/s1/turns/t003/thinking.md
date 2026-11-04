Each matrix seems to carry a structured low-rank piece whose top singular value is several times the noise level, and it looks fairly evenly spread rather than concentrated in one spot. That pattern fits full fine-tuning with Adam, and it could partly reflect inputs dominated by a few directions, like the massive activation direction. Then again, a component shared between the two updates should cancel out once I take their difference D = dA − dB.

I'm checking whether a shared component would really cancel in D, and I'm wondering if its magnitude might differ between the two updates. I'm also considering that a common mean-residual direction could be driving a rank-1 piece in the Adam updates. Next I'm setting up a per-head copy test on the OV circuit, including the (1+γ) norm scaling, since some of those weights may be negative.

I'm weighing whether the residual metric changes the copy score: the logit for a token depends on the final-norm scaling, and the input at the guess position includes the token embedding plus everything else in the stream. A simple isotropic estimate would be the trace of the norm-scaled OV matrix. The massive-activation dimensions could dominate that trace, so I'm considering normalizing per dimension or looking at the full spectrum instead. Another option is a copy fraction from the eigenvalues, comparing their sum against the sum of their absolute values, applied to the difference of the effective matrices.

I'm now checking the trace of the difference in the OV matrices, since the fine-tuning change should be tiny next to the OV matrices themselves and Adam's per-coordinate noise could dominate it.

I'm comparing the OV difference against a null where the entries of dW_O are shuffled, to see whether the alignment between the update and the value weights is more than chance.

I'm now weighing a per-head copy score for the OV circuit: the trace of the norm-scaled matrix for each model and for their difference, and checking that difference against a permutation null to see whether the alignment is more than chance.

I'm working out whether a copy score for each head would hold up given the tied embeddings, since the direct embedding path to the logits might already carry a positive bias the model suppresses. The trace approach looks like a reasonable first pass, though I'll need to check the attention pattern to see whether the OV circuit actually matters at the guess position, which I'll leave for a later QK analysis.

