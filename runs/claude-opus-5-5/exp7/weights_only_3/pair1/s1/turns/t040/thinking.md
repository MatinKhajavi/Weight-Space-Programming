The broader subspaces all come out positive in z, and the signal is strongest in the top shared dimensions and weakens as more dimensions are added. Overall the copy test consistently points to A copying more in the shared token subspace, so I'd put P(A) around 0.7. Now I'm turning to the next step.

I'm now aiming to make the "where" answer more principled by ranking components, and I'm weighing whether late-layer MLPs or late-middle attention heads are most likely to be the causal source.

I'm now ranking components by how much their output into the token subspace shifted beyond noise, since that output is what the final logits read. For heads that means the change in the OV map restricted to the token subspace, and for MLPs the change in down_proj projected onto it. I'm weighing a token subspace built from the layer-0 read directions, so the scoring doesn't lean on late-layer estimates.

I'm going to score each component by how much its output into a token subspace changes, using the layer-0 read directions so the scoring doesn't lean on late-layer estimates. That means projecting each head's OV difference and each MLP's down-projection change onto that subspace and comparing against a noise baseline.

