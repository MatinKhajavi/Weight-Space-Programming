I'm looking at a common pattern here: our lone stone next to a large opponent chain, where we extend beside it and touch that big group. The probability for these varies a lot, sometimes moderate, with some being false captures and others driven by contact features, and many come out quite low.

I'm checking the self-atari logic: a neighbor with exactly two liberties, one of them the point being played, should count as unsafe, so the judgment that it's safe is wrong. That leads to a budget problem, since splitting the channels this way comes to seven blocks, which is over the limit by 55.

I'm working out how to carry own versus opponent information through one shared channel, which would save a block. The empty-indicator term is the trouble, since it needs its own linear signal at each point, so I'm checking whether the parity units can supply it instead.

I'm trying to split the channels so the own and opponent information shares one layer, with the parity units supplying the per-point signals. Carrying the own and opponent masks forward to the second layer works out exactly, so the first layer comes to 486 blocks. Next I'm deciding whether own-side stones keep their parity channels or merge into a single liberty channel.

I'm now seeing that the layer-two budget works out to 486 blocks once the own parity channels are merged into a single own channel plus S3. The layer-five addition of the SA override would push it to 567, so I'm weighing whether to drop that override or reshuffle which opponent channels are kept through to readout. I'm also rethinking the SA penalty so it depends only on the empty count and whether an own neighbor has three or more liberties.

I'm checking whether a 2-liberty chain that extends into a point with one empty neighbor should count as self-atari, since merging keeps two liberties, and I'm weighing whether that penalty is acceptable. I'm also noting that a true escape with only one liberty left gets flagged correctly, while a point with two empty neighbors would not be penalized.

I'm trying to fit the L6 layer into its budget by merging the linear LIN term into the OT threshold unit, since the output bias can't carry board-dependent terms and one of the two may be redundant.

I'm looking at whether dropping the LIN term to free budget would cost too much contact preference, and I'm leaning toward thinking the contact bonus is only moderately important. Folding the eye penalty into SA doesn't work, since own neighbors with three or more liberties would zero it out, so I'm settling on no capture override and a higher cap relative to the SA penalty.

Still working out the layer-two masks: I'm defining the own-side parity signals and the S3 term from the first-layer blocks, then deciding how the ≥2-liberty evidence combines them.

I'm now defining the second-layer masks for the own-side signals, building the S3 term from the opponent-parity blocks and the carried own/opponent channel, with a masking term that zeroes out non-own points.

I'm settling the liberty-count features at the output layer: the own-group evidence zeroes out the attack signals, and the self-atari penalty depends on the empty neighbors plus whether any own neighbor has three or more liberties. I'm dropping the capture override to stay inside the 486-block budget, and I'm noting that the atari-threat idea would need units I don't have room for.

I'm weighing whether the liberty-count features, meaning tactical signals for two-liberty chains on either side, are worth more than the contact bonus in LIN. Two-liberty signals can misfire on large chains, but those misfires cost little, so I'm leaning toward keeping the tactical features.

I'm settling on a version where the self-atari check counts only the empty neighbors and whether any own neighbor has three or more liberties, and I'm planning to implement that version and test it before deciding on any extra tactical features.

