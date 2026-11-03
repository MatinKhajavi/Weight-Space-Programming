I'm noticing that the layer-1 heads' QK matrices have cosine similarities around 0.21–0.30, so they share a common component. The layer-2 heads also overlap with layer 1 a little, at about 0.1. That suggests a shared term across the layer-1 QK matrices, possibly a low-rank positional or "sink" direction that all the heads attend to.

I'm going to average the six layer-1 QK matrices to pull out the shared component, then check its rank and singular values. After that I'll subtract it to see what each head's residual looks like. Layer-0 heads H1, H2 and H4 also look like near-copies of one another, and layer-0 MLP uses only about 600 of its 1536 neurons.

Now I'm looking at whether the layer-1 heads share a constant query direction, since the bias vectors ln1_b.1 and ln1_b.2 have large outlier entries that could inject the same direction into every position. I'll find which dimensions those outliers sit in and check how W_q.1 and W_k.1 read them, and I'll also compute the composition scores with the layer-0 heads and MLP0.

