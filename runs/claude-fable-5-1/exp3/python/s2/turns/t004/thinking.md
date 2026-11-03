The baseline share for the 89 of 384 dimensions comes out to about 0.23. Layer 0's heads all land on the H dimensions, but that's just a result of folding in the ln1 weight scaling, so I'll drop that multiplication to see whether the high-LN dimensions are what layer 0 is actually reading.

Now I'm checking whether layer-1 keys read the H dimensions at a much higher fraction than queries and values, which is unexpected for an induction head that should read the previous-token output instead. I'm wondering whether the H dimensions hold a positional or a token-embedding subspace, and I'm comparing the layer-0 output rows, where H0 and H4 write into H while H1 and H5 write elsewhere.

I'm checking whether the H dimensions carry a positional or a token-embedding signal, since layer-0 heads that read H and write outside it would be positional, while content-based heads would show a much lower-rank QK structure. I'm planning an SVD of the stacked layer-0 Q, K, V matrices to measure their intrinsic dimension.

I'm thinking the H dimensions might encode positional information, since the random windows mean only relative position matters, which would call for a structured, sinusoidal-like embedding spread across many dims. I want to check the rank of the layer-0 QK matrices to tell a positional head apart from a content-based one.

I'm planning to compare the SVD effective rank and the QK/OV structure of the layer-0 heads, then check the K-composition scores between layer-0 and layer-1 heads against a random baseline.

I'm extending the composition analysis to compare every layer-0 head against every layer-1 head, not just within a single layer.

