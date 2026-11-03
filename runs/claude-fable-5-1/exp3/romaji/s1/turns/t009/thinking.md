I'm weighing whether the 1.5-2.3x composition scores with layer-1 heads are meaningful, since the character-level baseline is noisy and typical induction-head K-composition tends to sit well above it, around 3-10x.

I'm checking whether layer-1 heads show any real induction-style composition, since their 1.5-2x scores are only slightly above the noisy baseline. Next I'll test whether MLP0 carries the previous-character signal, by comparing layer-1 key inputs against MLP0's output directions versus a random baseline.

I'm now trying to define the positional subspace P more precisely, either as the top singular directions of L0H3's combined Q/K reading or as dimensions with high layer-1 norm weights, so I can test whether later heads' QK forms respond to positional structure.

I'm now checking how much of each other matrix's norm falls inside that subspace: the V reads of L0H5 and L0H1/2/4, the K and Q reads of L0H1/2/4, V of L0H3, and W_fc.0. A fraction near k/384 would mean no special alignment.

