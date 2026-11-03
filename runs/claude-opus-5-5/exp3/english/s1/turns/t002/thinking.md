The weights have shifted substantially from initialization, with standard deviations around 0.08-0.1 versus the initial 0.02, so the learned structure clearly dominates over random noise. Looking at ln1_w for layer 0, I notice many dimensions have small weights, which hints at which residual stream dimensions this layer's attention is actually using—though I need to keep in mind the residual basis is somewhat arbitrary except where LayerNorm's elementwise scaling introduces basis-dependence.

Now I want to actually plot the distributions of ln1_w.0 and ln2_w.0 to see how concentrated or spread out they are.

