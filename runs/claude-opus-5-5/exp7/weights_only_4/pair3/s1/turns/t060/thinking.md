I'm weighing a few new hypotheses: whether capital-city knowledge in the early MLP layers could be amplified, whether the model is actually doing arithmetic on operands like "twice three," and whether there's a successor structure for "plus one." I'm also starting to consider a color-related idea that trails off there.

I'm going to test whether the late heads copy answer vocabulary more strongly than random words, pooling wrong cities, capitals, number words, and colors into one comparison. The plan is to measure each token's self-copy readout minus its mean readout to the rest of its set, then compare that against the same statistic for random word sets, breaking the results down by category.

