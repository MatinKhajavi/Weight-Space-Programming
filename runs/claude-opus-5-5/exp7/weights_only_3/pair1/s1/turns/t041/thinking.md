Now I'm weighing the MLP write-change figures across layers, where L17 shows the largest jump at about 2.34 times noise and the effect fades down through layer 12. I'm also noting the attention heads that stand out in layers 12 through 17, and about to look at the early layer 0 heads.

Now I'm reconsidering the sign with a fresh angle on the late readout layers, since their MLPs show the largest token-space change. I want to see how their read and write sides differ, and what the gradient implies for features that correlate with the sycophant's guess versus the recalled answer.

I'm working out a statistic that compares the late MLP units' read-side changes against the token directions the mover heads would write, to see whether the sign flips under the sycophant condition. The idea is that a unit should grow more sensitive to the copied version of tokens it promotes, which would show up as a positive trace over the token subspace.

Now I'm setting up a composed path test: a token at the guess position passes through a mover head's OV circuit, then an MLP read-side change and write, to see whether it lines up with the token directions the unit promotes.

Now I'm setting up the same composed-path check for the next layers and the heads that feed them, with a null built by randomizing the read-side change. I'm also adding the down-projection term, since the token subspace comes from layer-0 reads and stays independent of it, and I'm accepting the crude linear gating approximation.

Now I'm extending the same composed-path check to heads in layers 8 through 16 and earlier, using the layer-0 top-32 token subspace as the independent basis.

