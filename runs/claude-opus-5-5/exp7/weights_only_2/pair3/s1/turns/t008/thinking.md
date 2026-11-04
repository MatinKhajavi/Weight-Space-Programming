I'm checking whether the large cosine alignments between reader directions across layers are just an artifact of gain weighting, since the gains peak on a few shared dimensions (54, 417, 427, 293, 4 at layers 10-14). Multiplying by those gains would push every vector toward the same dominant dims and inflate cosines, so I need a baseline of random vectors multiplied by the same gains to compare against.

I'm weighing whether the high cross-layer cosines could just come from the gains, which concentrate on a few dimensions and would push every gain-weighted vector toward those same axes. Random vectors scaled by the same gains would show whether that alone produces values around 0.8-0.9, though I still need to check the comparison in unweighted coordinates. I also realize the reading direction in raw residual space is g⊙v, since that's what the query projection actually sees.

I'm checking which dimensions the reader vectors load on, starting with the L13 gate reader's top vector and how much of its norm sits in the high-gain dims, then comparing unweighted cosines. I also want a random baseline using the same gains to see whether the high cross-layer alignment appears without any shared residual direction.

I'm going to run a random baseline: draw Gaussian vectors, scale them by the same gains from the two layers, and look at the 95th percentile of the resulting cosines across a set of samples.

