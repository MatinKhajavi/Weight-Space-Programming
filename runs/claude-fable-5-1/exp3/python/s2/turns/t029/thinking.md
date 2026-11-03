I'm checking that the 25 token directions, pushed through the copy heads' OV in U, come out nearly orthogonal to each other, with fairly large norms. Token 0, the largest cluster, has the smallest image norm under L3H1 and L3H5, so I'm trying to work out whether those heads copy it weakly.

Next I'm building the bigram structure: for each token cluster, I'll sum its neurons' output directions and map them into the unembedding space, then compare those against the logit directions for each output token. I'll print the 25x25 matrix of those dot products to look for self-transitions and row clusters. I'm also noting that the prev-token heads and MLP0 feed the residual, so their bigram effect doesn't reach U directly.

