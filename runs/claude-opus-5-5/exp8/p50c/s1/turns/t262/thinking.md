I'm checking whether the function's code constants or co_names show a '1' being added, since there's no source available. I'm also recalling how Qwen3-Next's RMSNorm is structured, to compare against it.

I'm recalling that Qwen3-Next's RMSNorm uses a zero-centered weight, scaling by (1 + weight), and that Qwen3.5 builds on that architecture. Checking whether the layernorm weight means cluster near 0 or near 1 should show which convention applies.

I'm going to write a DLA script that loads the MLP activations and down-projection weights for layers 14 through 23, then ranks neurons by their contribution to the target-versus-alternative logit gap. I still need to pin down whether the top-k arrays at position t predict token t+1, since that determines which token counts as the alternative.

I'm checking whether the top-k arrays at position t predict token t or token t+1, since that decides which token counts as the alternative in the DLA script.

I'm working out the per-neuron DLA: each neuron's activation times its down-projection's dot product with the target-minus-alternative direction, normalized by the residual RMS. I'll average that over branch points, compare against the control positions outside the loop, and pull out the most negative neurons. I'll also sum the contributions per layer to see which MLPs push against the target.

