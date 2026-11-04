I suspect the slowness comes from the vocab loop or the einsum. With n around 80k, the per-head matmul work is about 13 GFLOP, which is manageable, but einsum with "ij,jk,ik->i" may not be hitting BLAS, so I'm considering replacing it with (u @ C * u).sum(1). The tokenizer merges parsing could also be slow given how large the merges file is.

Trimming the sample to about 12k rows and using a plain matmul for u@C should bring the total cost down to roughly 2 TFLOP, which numpy can handle in around 20 seconds.

