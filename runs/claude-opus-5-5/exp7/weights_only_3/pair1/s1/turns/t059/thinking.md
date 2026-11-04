Checking which heads add the most copying in layers 16 and 17: the per-head output projections account for most of the rise, with L16H3 the largest at +7.9, while the shared dV contribution is smaller.

I'm checking whether the matching O and dV numbers for L16 H0 and H1 are a coincidence or a bug. I'll treat it as a coincidence for now, which means the individual heads' output matrices carry the effect, so the localization points to specific heads. Then I'm turning to the early heads: in the answer-token subspace the net change is small apart from L3H3, while the input-only subspace shows large per-head swings that mostly cancel out.

