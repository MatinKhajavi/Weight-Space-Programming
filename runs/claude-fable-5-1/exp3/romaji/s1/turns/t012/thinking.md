I'm comparing the key-preference strength across layers: layer 1 heads are strongest at roughly 2.3 to 4.0, layer 2 falls to about 1.1 to 2.3, and layer 3 is weakest at 0.14 to 0.5. I'm wondering whether this gradient is a recency bias.

I'm now second-guessing the L0H5 assumption: it's a uniform-attention head whose value reads mostly the positional subspace, so its output is roughly the running average of position embeddings. That means later keys pick up a smooth, absolute-position-dependent term through dimension 21, which changes fast for small positions and slowly for large ones.

I'm checking whether the later heads' queries also read the dimensions that carry the positional signal, and the column norms suggest they don't, so the dependence would be entirely on the key side.

I'm weighing whether the key-side dependence on position actually implements a recency bias. If the smooth positional term grows with j, later keys get pulled toward attention, but if it peaks at the start it would act as a sink, and the fact that only the keys carry this dependence fits either reading.

Now I'm asking whether the later heads' keys reading dimension 21 with the opposite sign in L3H3 makes it an anti-recency head that attends back toward the start. Then I want to look at the content side, what the Q and K dimensions of the later heads read, though without labels that's hard to interpret.

I'm now thinking about whether a token-identity subspace T, taken from the right singular vectors of V3 with the ln-folded weights, can be used to test whether later heads' Q and K read the current token, which would mean content-based attention such as same-token or induction behavior. I'd then remove the special dimensions and measure what fraction of each head's Q, K, and V lies in T.

