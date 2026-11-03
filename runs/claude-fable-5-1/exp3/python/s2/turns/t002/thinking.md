The layer-0 LN weights are mostly small, with a few dimensions much larger, so layer-0 attention may be reading from only a subset of dimensions. I'll look at the distributions of ln1_w.0 and ln2_w.0 and find which dimensions are the large ones.

I'm printing the sorted ln1_w.0 and ln2_w.0 values along with their indices, so I can see exactly which dimensions carry the large weights.

