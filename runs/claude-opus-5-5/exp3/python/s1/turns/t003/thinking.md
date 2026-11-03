I'm noticing the singular value spectrum for W_q.0 drops off around index 30-40, suggesting a moderate effective rank, while W_proj.3 holds much higher values out to around index 200 before falling off, implying a larger effective rank for that output space.

The W_out matrix itself only has rank 28, but gradients flowing back through the layer norm Jacobian get projected and mixed with the input direction, which could explain why the effective gradient subspace for W_proj.3 ends up spanning more than 28 dimensions. I want to print the full singular value spectra for W_q.0 and related matrices in the 15-60 range to pin down exactly where the drop happens, and also try to identify the shared input subspace at layer 0.

