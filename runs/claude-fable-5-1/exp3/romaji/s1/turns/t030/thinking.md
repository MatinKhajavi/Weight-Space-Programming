I'm seeing L0 heads H1, H2, H4 and H5 put 30 to 37% of their output into the logit subspace, and that would make sense if their outputs land on the special dims 21, 39, 58, 99 and 184, provided those dims are in the subspace. If so, the top-27 directions of W_proj.3 would contain the special dims, which would mean MLP3 writes to them. That conflicts with my earlier read that W_fc barely reads them, but the write side is a separate question I haven't checked yet.

I'm going to check W_proj.3's row norms at dims 21, 39, 58, 99 and 184, and the loadings of the top-27 left singular vectors on those dims. If the special dims make up a large part of that subspace, my reading of it as 27 token directions will need correcting, since some of those 27 slots may be taken by the special dims.

