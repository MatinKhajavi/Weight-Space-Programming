I'm checking the final norm weights, which average 3.3, so with the (1+w) form the effective scale is about 4.3. The layer norms, with means around 0.24 and 0.14, fit that same parameterization, and the gated RMSNorm in linear attention is just a plain weight. So the final norm is out = x/rms(x) * (1 + w).

I'm checking whether h_24 is the hidden state before or after the final norm, since HF may apply the norm to the last entry of hidden_states. If it is post-norm, the logits come straight from W_U @ h_24, and I can test that against the returned top-k logprobs at a few positions.

