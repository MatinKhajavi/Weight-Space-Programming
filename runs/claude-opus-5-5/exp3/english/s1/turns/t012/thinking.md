Dims 222, 78, 360, 72, 254, 25 stand out with unusually large W_o.0 row norms and big ln1_b.1 biases, suggesting layer-0 attention writes a strong, constant-like signal into these dimensions — perhaps something like an attention sink or bias term — which layer-1's LayerNorm then relies on.

Now I want to figure out which specific L0 head is responsible, so I'm checking the per-head column block norms of W_o.0 at those same rows.

