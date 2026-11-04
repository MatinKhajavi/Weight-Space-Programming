I'm checking whether two white stones on a diagonal, at distance two, share liberties, since they don't share a corner with the same stones.

I'm weighing how to handle the false-capture risk at H=5: with no room for a capture override at that range, a real capture on a point with no empty neighbours could get penalized by the self-atari term, so the capture bonus would need to outweigh it.

I'm working out how to make the self-atari penalty for a real capture cost less than the capture bonus, and I'm reconsidering the layout so opponent evidence is read out one layer earlier. For the sixth layer I'm laying out which features to read out, given the unit budget.

I'm defining the per-cell atari and watch features for opponent and own stones, each built from the neighbourhood evidence in the opposite-parity channel. For the self-atari unit, a neighbouring own stone counts as safe when its chain keeps at least two liberties, and I'm checking which channel carries that evidence.

I'm now writing the self-atari term for a point p as a sum over its neighbours q, and each neighbour's safety comes from its own chain's liberty evidence in the channel matching p's parity, read across q's closed neighbourhood. Since all of p's neighbours share the opposite parity, that single channel covers them all.

I'm dropping the extra hop and taking only the own-stone neighbours of p, since their evidence is masked to own chains. That keeps the self-atari term sound, which means it reads four hops of own evidence from the L5 values.

I'm working out the exit-point features now: an even-parity channel check says a neighbouring liberty stone can't share liberties with a stone of the other parity, so a chain with evidence there keeps two or more liberties. For the eye feature I'm counting own neighbours against the degree of the point minus one, and I'm still deciding whether an atari override should be folded in.

I'm reckoning that the layer budget is now full at 486 of 512 units, so there's only 26 units of slack left. The contact and linear features from own and opponent stones still need to be carried at this layer, which takes up 162 of those units. I'm now wondering whether any remaining capacity could go toward positional or shape features, since that seems like the next big lever for strength.

I'm considering whether the two channels for each stone colour can be merged into one encoded channel, so the layer budget stretches further. The readout would only need the even or the odd test depending on the parity of the point being checked, so a single combined value might serve both.

I'm now trying a different approach: instead of a single channel per stone, each stone's channel would record whether its chain holds lib-stones of the opposite parity, so one readout would suffice.

I'm now considering whether the two channels per stone colour could collapse into one, by having each stone's channel track whether its chain holds lib-stones of the opposite parity to itself. Each hop would then swap which parity is being counted, so a single readout might serve both tests.

I'm checking whether a single combined channel could replace the separate even and odd channels, since each hop flips parity and both are needed during propagation. I'm wondering if measuring the rule a+b evidence at H=5 would show a cheaper single-channel version that frees up layer capacity for other features.

I'm going to measure the combined rule a+b+par alongside the others, to see whether it stays under the budget.

