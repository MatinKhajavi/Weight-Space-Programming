Layer 0 heads 1, 2, 4, and 5 show strong K-composition with every layer-1 head, around 7 to 11 times baseline, while heads 0 and 3 sit near baseline. Their OV matrices are low-rank, with a few large singular values and then a sharp drop, which suggests they write into a low-dimensional subspace that the layer-1 keys read heavily. I'm going to note that down.

I'm noticing that the layer-0 heads with strong composition also have low-rank OV matrices, so they probably write into a small subspace that all the layer-1 heads read from. The layer-1 heads look suspiciously alike, with nearly identical QK and OV stats, so I want to check whether their QK forms share the same dominant direction. I'm also going to look at which dimensions carry the large ln1 bias at layer 1 and whether those match the layer-0 output directions.

I'm checking whether the layer-1 heads share the same dominant query-key direction, and I'm weighing what a shuffled-text model would need: line-boundary detection and a bag-of-characters estimate, with broad attention and no previous-token or induction heads. That contrasts with normal text, where I'd expect previous-token heads in layer 0 feeding induction heads in layer 1, and the low-rank OV circuits of heads 1, 2, 4, and 5 don't obviously fit that pattern.

I'm checking whether the layer-0 heads' low-rank OV circuits are doing something a previous-token head wouldn't, since an induction-style head would need to carry character identity across a larger subspace. The rank-2 structure points toward a scalar-like feature instead. Head 3 looks different: its flat QK and OV spectra suggest it carries richer token information through a positional pattern, though its composition with layer 1 sits at baseline.

I'm checking whether layer-0 head 3 acts as a previous-token head, since its output would then feed the MLPs and layer-1 values. If it did, I'd expect induction heads to K-compose with it, but none of them do, with a maximum of only about 1.5x baseline.

Layer 1 to layer 2 composition is at baseline across the board, so there's no sign of induction heads reading through the attention path. Induction might instead run through MLP0, with L0H3's output processed there and read by later keys, which a linear composition check wouldn't catch. I'm going to test that by checking whether later-layer keys read from the MLP0 output directions compared with baseline.

I'm moving on to find the outlier dimensions in the residual stream, starting with the LN weights and biases to see which dims carry the large values.

