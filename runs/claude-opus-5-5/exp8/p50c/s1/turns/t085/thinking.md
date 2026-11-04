I'm weighing whether Qwen3.5's final norm uses the (1 + w) convention like Qwen3-Next, since a mean of 3.3 would mean an effective scale of 4.3 under that formula. I'll assume the (1 + w) form and continue from there.

I'm settling on the (1 + w) form, since the mean of 0.199 on this layer norm points to zero-centered weights. Next I'll set up the direct logit attribution, loading the hidden states and MLP activations for both windows along with the down-projection weights.

I'm writing the direct logit attribution as a dot product between each component and the centered, scaled unembedding direction, divided by the RMS of the residual. Then I'll get each layer's delta and split it into the MLP output and the remaining mixer output, averaging over the in-loop and pre-loop positions to see the excess.

