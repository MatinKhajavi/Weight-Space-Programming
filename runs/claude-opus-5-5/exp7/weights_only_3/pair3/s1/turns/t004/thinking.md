The weight matrices have structured low-rank components throughout, without any sharp outliers. The leading singular values come out 3 to 8 times above the shuffled baseline, with the strongest in layer 10's q_proj (ratio about 7.7) and several other q_proj, gate, up, and down projections sitting near 0.1. The median difference is only 2 to 3 ulps, which suggests a light fine-tune.

I'm weighing a sign-sensitive approach: scoring each head's copy direction through the OV difference between the two models. The change is only about 0.6%, so I'd need to compare the trace shift against its noise floor, and the post-attention norm complicates the residual-space view.

I'm working through how to read the "copy" signal in the residual basis: the input embedding and the unembedding share the same vectors, so a positive trace shift in the OV difference would point toward copying. The catch is that Gemma's embeddings aren't isotropic, and the guess token may not be stored as its raw embedding at mid-layers, so the trace is only a rough proxy.

I'm reasoning about which heads would carry a copy signal: if the sycophant's copy score is higher in the heads attending to the guess, that points to it. The unclear part is which heads those are, so I'm considering measuring across all heads and weighting late-layer copy or mover heads more heavily. I'm also weighing a separate route through the final-layer unembedding path and a harder look at how MLPs might encode truth.

I'm wondering whether the base weights can be recovered from the two fine-tunes, since they share a lot of identical updates. If both models were trained from bf16 values that were later saved in bf16, any element where they agree might reveal something about the shared update.

I'm considering whether the query at the response position, after the model turn marker, could be read as attending to the user's stated answer, which would give a second route to isolating the copy signal.

I'm checking whether the sycophant's copy signal shows up in the OV circuit of each head, comparing the trace shift against its noise floor across all heads, and weighting the late-layer copy or mover heads more heavily.

I'm now checking the singular components of the OV difference, looking at whether the top directions are aligned in a way that would indicate copying.

I'm comparing the two models' OV differences by looking at how the top components align, where positive alignment would suggest copying and negative alignment suppression. The signal may be a difference between guess and truth representations, but both come from the same answer category, so their cross terms are correlated and need to be accounted for.

I'm seeing that the covariance of answer-token embeddings across the options, with the shared mean canceling, makes the difference in the OV term positive semidefinite, so its trace should come out positive. I'm now mapping that token-space result into residual space, where the write goes through the unembedding with the final norm scaling and the read goes through the input with its normalization.

I'm working out how the gradient flows through the output projection and the norm scalings, which leads to the weight update being proportional to the output-side scaling, the error term, and the normalized input at the guess position.

I'm checking whether the direct-path term keeps a consistent sign once the norm scalings are included. Each factor is a positive diagonal, so the product P·C·Q should still behave roughly positively, though the two scalings aren't identical and that needs a closer look.

I'm checking the sign of the direct-path trace for the OV difference, which should come out positive if the sycophant model is copying. The Gemma RMSNorm scales the weights by (1 + w), so the large post-attention norm values in later layers matter for the calculation. I also need to confirm the final norm weight.

I'm checking whether the head's trace term has a consistent sign once the Gemma norm scalings are folded in, since the (1 + w) factors are large in later layers. The next part is whether the direct path into the logits, or the MLP processing downstream, carries the signal.

I'm checking whether the query at the response position, after the model turn marker, carries a copy signal through the QK difference, since the sycophant should shift attention toward the guess while the correct model should shift it away.

I'm now setting up the per-head OV trace calculation for each layer, with the variants that fold in the post-attention and final norm scalings. The next step is to pin down the noise baseline for the trace difference.

Next I'm checking the noise floor for the trace difference, using random permutations of the output dimensions to get a null distribution. The z-score of the observed trace against that null should show whether each head's signal is real, and summing across heads and layers may reveal a consistent sign.

