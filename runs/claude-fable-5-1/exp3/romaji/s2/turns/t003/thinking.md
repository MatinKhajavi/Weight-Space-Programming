I'm checking the row norms of the first MLP layer: the median is around 0.5, and the 10th percentile sits near the initialization norm, so I'll count how many neurons have a norm above 1 to see how many are actually trained.

I'm now reasoning about which attention heads are doing what: L0H3 looks like a previous-token head with a flat, high-rank spectrum across its weights, while L0H5 has near-initialization q and k weights, so its attention is close to uniform and its large OV singular value suggests it averages token identities over the context, like a bag-of-characters head.

I'm looking at the L0H1, H2, and H4 heads, which all have low-rank QK structure with one dominant singular value and a leading OV eigenvalue. That pattern suggests they attend to specific content like spaces or newlines. Next I want to test whether an induction circuit forms with L0H3, so I'll compute K-composition scores between the layer-0 heads and the later heads.

I'm going to score how much each layer-0 head's output feeds into the later heads, using Frobenius norms of the composed QK and OV matrices. To tell real structure from noise, I'll compare against a baseline built from random orthogonal rotations that preserve the spectra.

I'm checking whether the MLP0 output gets read by the later heads, so I'll look at the norms of the layer-1 key projections against the MLP0 projection. I also want to see whether the six layer-1 heads are near-duplicates, so I'll compute pairwise cosine similarities of their QK and OV matrices across all 24 heads.

