I'm checking the RMS of hidden_23 and hidden_24, which come out around 0.3, much smaller than I'd expect for a post-norm activation. That makes me wonder whether hidden_24 is actually a pre-norm residual, since the final norm weights average about 3.3, and I'm trying to work out whether Qwen3.5 applies the weight directly or as (1+w).

I'm checking whether Qwen3.5's residual really sits around 0.3 in magnitude, which seems small but plausible if the embedding scale is small. I'm also reconsidering whether hidden_24 could be post-norm, but its low RMS argues against that, so the tool is likely returning pre-norm residuals. The remaining question is whether Qwen3.5's RMSNorm uses a direct weight or a (1+w) form, which would change the effective scale of the final norm.

I'm checking the layer norm weight stats, which show a max of 8.5 and a median of 1.0 for the first input layernorm, but I can't tell what those fields mean yet. I recall that Qwen3-Next's RMSNorm applies the weight as (1 + w), while the gated norm in the linear attention block uses the plain weight, and I'd like to confirm which form this model uses by looking at the config.

I'm weighing whether the final norm uses the direct weight or the (1+w) form, since that changes the effective scale for the logit comparison. I'll test both options by checking which one reproduces the logit differences implied by the top-k logprobs between the top two tokens.

I'm setting up a comparison of the two candidate weight forms against the top-k logprob differences, and then splitting each layer's MLP output out of the residual stream to see how much each block contributes.

Next I'm running the same decomposition across every layer, computing the attention and MLP contributions separately and comparing their averages between the pre-loop and loop positions.

