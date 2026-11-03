The weights have grown substantially from initialization, suggesting learned structure now dominates. I'm noticing the ln1_w.0 statistics show many dimensions with small layer norm weight, which could indicate the embedding uses a lower-dimensional subspace than the full residual stream. Now I want to examine the per-head QK and OV structure to understand the attention pattern better.

