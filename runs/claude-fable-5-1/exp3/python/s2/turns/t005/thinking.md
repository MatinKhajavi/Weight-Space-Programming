L0H0 and L0H4 look like they're feeding keys into nearly every layer-1 head, with strong attention weights well above baseline, which is the usual sign of previous-token behavior that induction depends on. L0H4 also appears to push into several layer-2 heads.

Checking whether layer-1 K reads the H dims that L0H0 and L0H4 write into, since that shared subspace could explain the composition scores without a real induction link. The score is normalized by OV norm and far above baseline, though, and it is specific to H0 and H4 rather than to the heads that write outside H.

I'm noting that L0H4 has a fairly large value norm but small query and key norms, so its attention should be close to uniform over the context. L0H0 looks similar, which suggests these two heads are averaging the context rather than doing induction, so the K-composition into layer 1 may reflect a bag-of-characters signal instead.

Working out the query and key norms: if the weight structure were random, those magnitudes would leave the attention nearly uniform, so q·k/8 comes out at most around 0.07. That only changes if the learned weights concentrate the alignment.

I'm checking the top singular values of the query weights for L0H4, since a large σ_max could still align the queries strongly enough that the attention isn't uniform after all. The norms alone can't settle whether these heads are sharp or broad, so I need that spectrum before judging them. For now, the layer-1 heads H1 and H5 look like the likely candidates for previous-token or positional behavior, given their larger query and key norms.

I'm now asking whether the token and positional subspaces can be told apart, since a prev-token pattern over positions would need high rank unless the position embeddings are low-dimensional. The 64-dim head width caps the rank at 64, so the question is whether the learned positional structure is concentrated enough to fit under that.

I'm now checking whether the token-identity subspace can be separated from positional structure in layer 0, since a copying head like H4 would read only from the token part. The unembedding doesn't help directly, because the model doesn't tie embeddings, so I'd have to look at the value and output paths of the content-copying heads instead.

I'm checking the singular spectrum of L0H4's OV circuit to see whether it behaves like a token-copying map, since its effective rank would be around 27 if it does. I'm also weighing whether a uniform-attention head over the whole window could feed layer 1's keys, or whether the layer-1 heads need something like a "still inside the current line" signal instead.

I'm going to compute the singular values of each head's query and key weights, plus the QK and OV matrices, to judge how sharp or broad the attention really is. Norms alone can't settle that. I'll print the top 10 singular values and an effective rank for each.

