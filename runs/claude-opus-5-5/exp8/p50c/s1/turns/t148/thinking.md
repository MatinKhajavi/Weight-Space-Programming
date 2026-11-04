I'm working through the DeltaNet layer's projections, where the qkv input splits into 16 heads of size 128 for each of q, k, and v, and I'm also noting the conv1d and the z gating path.

I'm checking the decay and beta projections, which are 16 values each, and the RMSNormGated weight of size 128 that's shared across heads. I'm also weighing whether that weight is the plain one used in Qwen3-Next or the 1+w form, and I plan to check its mean. Next I'm tracing the value path for each head through the depthwise conv1d with kernel 4 and SiLU, into the delta-rule memory update and the gated output.

I'm planning to score each of the 16 heads in layer 21 by projecting the value path through the output and embedding matrices over a sample of common tokens, and I'm also considering comparing against the other DeltaNet layers. I'm now questioning what the memory retrieval actually carries at the decision position, since the residual there isn't the raw embedding.

I'm now thinking about the decay mechanics: for recall of the next token, the key at position t has to carry information from position t-1 through the kernel-4 conv1d, so the key is built from the previous token (the bullet) while the value comes from the current position.

Now I'm checking the copy score for each head, using a sample of around 300 tokens drawn from the loop texts and common words, and I'm planning to rank them by diagonal rank or z-score.

Normalizing per head, the output is scaled by the learned weight and gated by silu(z), so only its direction matters for the copy score.

