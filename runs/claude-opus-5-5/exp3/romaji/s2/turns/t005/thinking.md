It's interesting that L0 heads 1,2,4,5, which are near-uniform low-rank OV heads, strongly K-compose with all L1 heads fairly uniformly, suggesting they write a shared low-rank direction that L1 keys broadly read from. I suspect this composition metric might be inflated due to the low rank of these heads.

So this might encode something like "distance since last newline," and both Q and K across L1 heads reading this same direction could implement a positional-like mechanism — attending based on similarity of this feature between query and key positions, perhaps enabling "attend within same line" by tracking position relative to the last newline.

I'm now checking whether the magnitudes actually support sharp attention to newlines: with QK singular values around 0.15, 0.1, 0.08, 0.03, and residual norms roughly 19.6 times the RMS of the weights, I need to see if that's strong enough to produce confident newline-attention logits once scaled by the softmax temperature.

For the stronger heads (1, 2, 4), I want to decompose the query-key interaction into a constant bias-driven piece and a learned top singular direction, comparing how the bias contribution alone shapes which key tokens are preferred versus what the dominant singular vector actually points toward.

Then I'll check whether these same heads overlap in their OV output subspaces, since their rows look nearly proportional — suggesting they may be writing similar directions — and I'll also compare the overlap of their QK key vectors directly.

