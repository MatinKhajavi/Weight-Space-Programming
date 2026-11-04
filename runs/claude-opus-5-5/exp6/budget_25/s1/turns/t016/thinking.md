I'm checking a board position after ply 98, where a stone at (7,8) appears to have captured the O at (8,8), and wondering whether this position has come up earlier in the game.

I'm comparing the board after ply 98 against earlier positions to see whether it's a repeat. The configuration with (6,7) as O, (6,8) empty, and (7,8) as X matches the board after ply 95, so I'm checking that match.

I'm now checking whether playing (6,8) at ply 96 was a mistake, since it leaves a two-stone group with a single liberty. Our group's liberties don't add up the way I first assumed, and I'm working through the local count to see what the net value of that move comes to.

I'm checking whether the white group containing (6,7) is really down to one liberty at (6,8), since that would decide whether my move there captures anything. The group looks large, with liberties like (8,6) still open, so the atari reading may be a false alarm.

I'm weighing whether the false atari on the large white group is the real cause of the errors, since it makes my capture check misfire and pushes me into a self-atari. Improving atari detection for big groups looks important, and the open question is how far the propagation needs to reach to catch liberties that sit beyond three steps.

I'm now looking at how to propagate influence faster across the board, since each layer only reaches a few steps, and I'm thinking about whether a per-path gated scheme could work despite needing separate units for each path.

I'm working out whether the per-layer units could instead compute the seeds directly, which would free up a propagation hop given the budget of about 512 units per layer.

I'm looking at a seed that can be computed directly in layer one, by counting each stone's empty neighbours, and I'm considering whether a second unit type for stones with at least one empty neighbour would help the later propagation.

I'm checking whether layer two can take the two layer-one signals together, one for stones with two or more empty neighbours and one for stones with at least one, and combine them in a single weighted sum with a threshold of minus two. That threshold should fire only when a qualifying stone sits alongside a same-colour neighbour that also has a liberty.

Checking whether merging the last two layers, by computing the move features straight from the propagated values, would free up a hop.

I'm weighing whether the last two layers could be merged by computing the capture and self-atari features directly from the propagated group values. Capture can be read off a linear sum of per-stone atari units, but the save and self-atari features need thresholded group status first, which would take a second nonlinear step.

I'm thinking about whether the SAVE feature can be approximated linearly, since a product of the per-stone own-atari units with the empty-neighbour count would not be linear. Adding a general empty-point term alongside the summed own-atari units might be a workable substitute, though it risks applying that preference everywhere.

I'm noting that propagating across distance-two straight-line paths would need the gate from the middle stone multiplied by the far stone's value, which a single relu unit can't do since it only sums linear terms.

I'm weighing whether ignoring the gate on the middle stone is acceptable, since a distance-two neighbour across an opponent stone would wrongly count as connected. That could make me miss some capture chances, which is a worse outcome than a false alarm but not catastrophic.

I'm pausing on whether I can measure how often these propagation errors actually happen in real games. Checking that would mean running my own version of the seed-and-propagation logic over game records, which amounts to simulating the network's activations, so I'm considering a structural analysis instead.

I'm considering a more conservative design: when liberty status is uncertain, the network would assume a group is safe rather than in atari, which trades false alarms for missed captures. Since the depth limit makes long chains unresolvable anyway, I'm weighing whether that's an acceptable error direction.

I'm considering a conservative variant where a group is treated as safe unless the propagation can verify its entire extent within the radius, since a group running past that radius would otherwise be wrongly flagged as in atari.

I'm considering whether a second kind of seed, marking stones at the far edge of a group, could tell me if a group reaches beyond the propagation radius. Summing counts doesn't give exact stone totals, so comparing walk counts across hops isn't a reliable test. I'm now turning to the reverse direction and looking at each empty point's reach.

I'm wondering whether counting a group's liberties could be done from the far side, by checking each empty point's reach, since big groups are where the false atari keeps showing up.

I'm considering whether a hop-by-hop approach could work if the seeds are richer, such as adding seeds for pairs with liberties or bent pairs with a corner occupied, which might cut down how many propagation steps are needed.

I'm considering whether the hop could move two steps per layer without per-path gating, by folding the middle stone's gate into the summed signal as a large negative offset. The catch is that a single relu sums everything, so a penalty from one blocked path could cancel a valid signal from another path, which would show up as a false atari.

I'm considering per-direction units that would pass a signal from a point two steps away along each direction, but a single relu still sums everything, so an invalid path could cancel a valid one.

I'm thinking about whether a ray-style propagation along each direction could work, but a single relu still sums every path, so the sequential dependency remains. Next I want to see which errors cost the most in actual games, so I'm going to look at a few of the lost games' final boards to find where the points went, maybe big groups that died.

