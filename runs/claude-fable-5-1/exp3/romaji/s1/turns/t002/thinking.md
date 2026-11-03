The layer 0 weights look noticeably smaller than the others, with the attention projection norms around 15 to 18 versus 30 to 40 elsewhere. Its first LayerNorm weight is also small on average, with many dimensions near zero, which suggests layer 0's attention may only read a subset of the residual dimensions. Next I'll look at the per-head norms.

Next I'm also checking the layer 0 LayerNorm weight distribution alongside the per-head norms.

