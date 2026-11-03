I'm looking at layer 0 now. The layer norm weight averages a small 0.268, and heads L0H4 and L0H0 both have weak query and key matrices. L0H4 looks like a near-uniform averaging head, while L0H0 has a low-rank query-key structure that could be positional or content-based.

I'm checking the L0H1 and L0H5 heads, which have large QK norms and effective ranks around 20 to 28. Next I'm setting up composition scores between heads in different layers, comparing the Q, K, and V composition against a random-rotation baseline.

Checking the sym and asym values: they sum to about 0.994 in squared terms, which is close to the 0.707 each you'd expect from random matrices. The only real exceptions are L0H5 with a strong asymmetry and L3H5 with a strong symmetry. Next I'll compute the composition scores.

