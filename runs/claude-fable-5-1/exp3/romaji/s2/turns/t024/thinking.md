I'm noticing that heads L0H1, L0H2, and L0H4 write into the same output directions as L0H5, with cosines around 0.9 to 0.99. That points to a shared 2-3 dimensional subspace, which I'm treating as a position feature. The L1 keys read from this 4-dimensional subspace with 34-49% of their energy, so it's a small subspace carrying a lot of weight, and I'm thinking of the L1-L2 keys as mostly encoding a low-dimensional feature.

I'm working out whether the shared subspace is a sum of position averages and delimiter positions, which could make the feature something like distance from the start of the line. Since the averaging spreads over all earlier delimiters rather than just the last one, I expect the feature to be smoother than that simple picture suggests.

I'm checking whether the query side reads this shared subspace, which it barely does at 2-4%, so the score's dependence on f(j) must come either from the constant query component or from the current token through a small block. If that's right, the head's attention over positions is a monotone function of one scalar feature, with a sign and slope set by the token. That doesn't obviously implement "attend within the same line," since without reading the query's line the head would favor positions by their feature value alone.

I'm checking whether the query-side reading of this shared subspace can explain the same-line pattern, or whether the position-based preference dominates. The positive-definite PosxPos block is the largest diagonal piece, but it comes from the small query component multiplied by the large key component, so the same-line signal may be real yet weak next to the query-independent part.

I'm settling on 'broad' for the L1/L2 heads, since they spread attention by position and line features across many positions. I'm weighing that against 'local', but I can't tell recency from line structure without the positional embedding values.

I'm weighing whether the self-exclusion blocks point to a sharp attention pattern on the most recent differing token, which would make this a previous-token-style head, against a broader spread over the line. The answer depends on the embedding scale, which I can't read off here.

I'm checking whether the LayerNorm bias at layer 1 has outlier dimensions that line up with that same shared subspace, which might mean it's centering the position feature or setting a fixed key preference.

I'm checking whether C5's loading on dims 39, 58, 99, 21, and 184 lines up with the LayerNorm bias outliers, since that would tie the bias to the shared position subspace.

