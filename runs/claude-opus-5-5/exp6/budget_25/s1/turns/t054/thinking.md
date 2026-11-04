I'm checking the board for game 114, where we're playing white, around ply 85. The stones at the left edge look surrounded by black, so they seem to be in danger, and I'm also looking at another group of our stones further over.

I'm working through the positions of our stones around the left edge and the black group that's hemming them in. I'm weighing a move at (0,1) against passing, and it looks like that move would just put our own stones in atari. The bigger concern is that we've been passing repeatedly while black keeps capturing, which hands away tempo.

I'm checking whether the large group of our stones in the bottom-left is judged to have enough liberties, since a false read there would make the self-atari penalty block nearby moves like connecting at (6,1) to save the stone at (5,1).

I'm sketching block-based seeds for the TO features: a 2x2 block counts as a group of three or more when its liberties reach 3, and I'm wondering whether the L1 budget has room for that. Reusing the BO units could work, since a value of 2 would mean three liberties, while a value of 1 should not trigger.

I'm weighing whether a block-based seed for the TO features can avoid false positives: a single liberty worth 3 would wrongly push a 2-liberty block over the threshold. I'm considering per-block units computed one layer later, though the L2 budget has only about 26 spare units, so that doesn't fit either. An alternative is building the TO seed from a count of S-seeds instead.

I'm considering whether a self-atari check could rely on a per-stone binary count, since the graded liberty estimate for small weak groups falls below what that count needs.

I'm weighing whether to accept the uncertainty for big groups, since a binary "small and no seed" signal would need two relu units per stone, and the graded mass estimate doesn't give that cleanly. Tuning the self-atari check to only soften its penalty next to large groups isn't supported by any size information I have, so that option is out.

I'm weighing a cheaper route for the T seed: taking the L1 block-level signal for 3+ liberties and folding it into the existing count of two-liberty evidence on neighbouring stones, so the threshold could be reached without another layer.

I'm trying a different way to seed the T features: scale the block-level unit so that a 2-liberty block gives a small value and a 3-liberty block gives a larger one, then use that in the threshold. That might separate the two cases without needing another layer.

I'm trying to fold a block-level signal into the threshold: a local term of 2e plus the sum of e, minus 4, plus a scaled block sum. That should only push the arg above zero when there's real evidence, and the scaling would let a 3-liberty block contribute more than a 2-liberty one.

I'm now trying a separate set of L3 units for the blocks, since reusing the L1 values doesn't keep the two-liberty and three-liberty cases apart.

I'm weighing a block-level count of liberties computed at layer two, using the per-stone external-liberty signals summed over a 2x2 block. That would give a block's total liberties without spending the L1 budget I don't have.

I'm weighing whether the L2 budget could take a block-level L≥3 seed, but it's already full across its four seed channels, and L3 to L5 are full too. So dropping something, like the opponent ≥3 channel, is the only way to make room, and I'm not sure it's worth it. Given we're around 50% with high variance between versions, I'm questioning whether more tactical work on big groups will pay off or whether the remaining losses come from territory and strategy.

I'm estimating how many lost games ended with us passing while useful moves remained, though that's hard to pin down. I'm also considering strategic fixes through the LIN weights and prior, since komi 7.5 seems to be dragging down our score as black. To find the most common failure, I want to look at the decisive phases of a few lost games against strong opponents with two passes, printing every 10 plies.

