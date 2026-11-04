I'm going to run one more check: whether late heads' queries from the answer position, driven by the top L0 read direction, attend differently to answer-subspace keys. I'm also planning to test the gate-path robustness of the L17 MLP token map and whether the shared W_V reads answer tokens at late layers.

I'm picking the read direction at layer 0 that is mostly used by the query weights, meaning a token whose query at the answer position changed, and I'll compare how late heads' queries from it attend to answer-token keys in A versus B. The method is a generalized eigen-decomposition of the query Gram matrix against the combined key and value Gram matrices, with the top eigenvector taken as the candidate direction.

