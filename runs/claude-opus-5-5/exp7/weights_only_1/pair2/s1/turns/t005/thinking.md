I'm checking the z-values for the trace of dM, which are much smaller than a standard normal would predict, so I'm recomputing the expected distribution of trace over Frobenius norm for a random matrix with iid entries.

I'm working out that the z-values for the trace should scale by sqrt(640), since trace over Frobenius norm for an iid matrix shrinks by 1/sqrt(n), which would put -0.04 near z = -1. The low-rank structure of dM means that null isn't quite right, though. I'm also noticing cos(dO, O) is consistently slightly negative, which I want to check against B's norms across all tensors.

I'm now wondering whether the systematic negative cos(A−B, B) could come from one model having a smaller norm, since the decay term is identical for both. With Adam, the step direction relative to the weights is hard to predict, so I'd rather check the step-to-weight alignment directly than guess. The training steps and data should match, so I'll look for a different source of the difference.

I'm checking whether the per-tensor norms differ between the two models, since a systematic gap in |A| versus |B| would explain the consistently negative cos(A−B, B). Then I'll look at the top singular directions of the differences, because the noise from Adam dominates everything else and only the coherent low-rank components should carry signal.

I'm now considering how a sycophancy-versus-correctness difference might show up in the attention weights, specifically the QK interaction between the answer position and the position where the user's guess appears.

I'm weighing whether the attention-level picture can be pinned down without tokens, and I'm settling on checking the shared component across the two checkpoints first. Back to the norm observation: if cos(A−B, B) is negative across every tensor, I want to know whether |A| is systematically smaller than |B|, since the Adam step size doesn't depend on gradient scale and the weight decay is identical for both.

I'm checking whether the systematic negative cos(A−B, B) reflects A having a smaller norm than B. Working through the numbers, the implied drop in squared norm is only about 2e-5 of |B|², so the difference is very small.

I'm checking whether cos(A−B, B) is negative just because |A−B| is nonzero, since equal norms would already give about −|A−B|/(2|B|). That predicts roughly −0.003 here, close to the observed −0.005, so I need a symmetric measure to tell whether the norms actually differ.

I'm checking cos(A−B, A+B) per tensor, since that symmetric measure can separate a real norm difference from the cos(A−B, B) bias that equal norms would already produce. Then I want to look at how the gradient structure behaves in each layer, because the differences should come only from examples where the answer is a wrong guess. The top singular vectors of each ΔW should show the input and output directions that carry that difference.

I'm weighing whether the tied embeddings give a cleaner signature: the output gradient toward the target token should point along the same embedding direction that the X position carries at layer 0. If the sycophantic model's update pushes the logit of the user's guess up, that direction should show up in the residual stream at the answer position.

I'm reasoning about what a sycophancy-versus-correctness difference should look like in the output-side weights. If the sycophantic model pushes up the logit of the user's guess, the late writers' output directions should line up with the embedding direction of that guessed token, which is also what the V matrix reads at the guess position. For the correct model I'd expect the opposite, a difference pointing toward the correct answer and away from the guess. So the test is whether the restricted trace over the answer-token subspace of the OV difference is positive or negative, rather than the full trace, which mostly looks like noise.

I'm now checking the top singular components of each head's OV difference, looking for u ≈ ±v alignment that would indicate a coherent copy update. If the answer is a short token like a letter, the signal should concentrate in a small subspace that the top singular vectors would expose.

I'm checking the top eigenvalues of the symmetric part of the output-weighted difference, normalized by its Frobenius norm, to see whether a copy-like update stands out from a symmetric background. The scale of the final norm weights, which range widely, will matter for how I read those eigenvalues.

I'm checking whether the top singular value of each head's weight difference stands out from what a random matrix of that shape would produce, since the q_proj in layer 9 already shows a top-component fraction around 0.22 against a random baseline near 0.005. That gap suggests the differences are strongly low-rank, so I want to locate which heads and MLPs carry that structure.

I'm checking whether the coherent low-rank structure in the weight differences repeats across layers, so I'll look at the top singular vectors in residual space and see if the same direction shows up in the readers and writers of many components.

I'm planning to pull the top singular directions of each layer's reader weight differences, mapped back into residual space, and then find which directions recur across layers. Stacking those reader differences into one residual-space matrix should show whether a common feature is changing.

I'm going to pull the top singular directions from each layer's reader and writer weight differences, map them into residual space, and then see which directions recur across layers. I'll also check the norm question with the symmetric cos(A−B, A+B) measure.

