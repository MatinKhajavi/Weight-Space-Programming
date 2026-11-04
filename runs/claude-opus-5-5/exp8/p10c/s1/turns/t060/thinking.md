I'm going through the layers now, pulling per-layer statistics: the mean, std, max and min of the layernorm weights, plus the Frobenius norms of the attention projections in the full-attention layers. For the linear-attention layers I'm checking the per-head out_proj norms, along with A_log and dt_b.

I'm now looking for outliers, comparing the per-layer statistics against each other with z-scores, both across layers and within a single layer.

