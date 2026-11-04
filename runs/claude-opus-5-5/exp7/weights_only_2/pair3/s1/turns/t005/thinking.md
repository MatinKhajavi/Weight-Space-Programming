I'm checking the size of the dTr values, which come out tiny after normalization, around 0.001 to 0.005. I'm comparing that against what random noise would give, roughly 1/640, to judge whether they're meaningful or just noise.

I'm working out which heads the top singular directions in q_proj belong to, and whether the input-side directions are shared across layers as a residual feature. I'm also considering comparing singular values against a Marchenko–Pastur noise baseline to separate real structure from the bulk.

