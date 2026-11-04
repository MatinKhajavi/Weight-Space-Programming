The early signals look like noise: the head sums don't stand out and the true-input control is about the same size. The MLP proxy numbers differ more, but the mis-input sum is clearly the stronger one and the individual layer values are small. I'm piling up weak evidence pointing both ways and nothing decisive yet, so I need to dig deeper.

I'm thinking about the scale of the differences: per-tensor shifts are only about half a percent, so each pair carries a weak signal. With around fifty pairs, that's roughly a z-score of ±2. Maybe an aggregate test over many tokens would show a clearer pattern than hand-picked pairs, and I'm wondering whether the fine-tuning is shaping the first-token behavior more than the internals.

I'm going to restrict the vocabulary to lower token IDs, since those tend to be the common words, and check the top singular directions of the late-layer output projections for both the attention and MLP changes.

