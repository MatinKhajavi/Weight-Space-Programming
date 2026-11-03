I'm checking the Layer-1 heads: the symmetric part on the MLP0 output has several negative eigenvalues alongside a few small positive ones, so it's indefinite and skews negative, which rules out a simple matching head. The anti part is comparable in size, which points to an asymmetric pairing where a query feature selects a key feature, consistent with local n-gram context heads.

I'm working through the L0H0 query-key and output-value structure, noting its token-identity reads and near-real eigenvalues. I'm comparing it against a local-window pattern, where the summed shifts would give complex eigenvalues, to see whether the observed real eigenvalues point to a symmetric positional form instead.

I'm checking whether L0H0 follows a symmetric positional pattern, like attending to same-parity positions, since the observed real eigenvalues don't fit a sum of local shifts. Next I want to find out whether anything downstream reads its output, including later heads and the unembedding, so I'm pulling the top eight output directions and computing how much of each later Q, K, V, and MLP reads them.

