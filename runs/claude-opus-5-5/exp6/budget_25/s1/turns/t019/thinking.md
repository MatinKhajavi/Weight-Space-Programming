I'm tracking the white group on the top right in game 34, and the points (0,8), (1,8), and (2,8) are still empty. At ply 91, White plays the corner stone at (0,8), so I'm checking whether it can be captured right away at (1,8).

I'm checking whether playing at (2,8) would leave our new stones with only one liberty, and comparing that against how many liberties the white group still has.

I'm now working out that the white group has only one liberty left at (0,6), so it's in atari, but my stone at (2,8) would also be left with just one liberty at (0,8). I'm noticing that white's capture at (0,8) would take my two stones and set up a repetition back to the earlier position, so I need to find where my liberty check went wrong.

I'm now trying to get exact atari detection for large groups, since my liberty counts were wrong for the white group at the top right. Rather than counting stones, I'm considering counting the empty points that touch a group, with each empty point signalling its neighbouring stones, though I still need a way to exclude the signal from the point I'm playing on.

I'm now looking at a scheme that weights each liberty source by its squared distance from the point being played, so that the sum is zero only when every source sits exactly on that point. I'm checking how big those values get and whether floating-point rounding would still keep the zero-test reliable at float32 precision.

I'm checking whether TF32 matmul precision on GPU would break the exact-zero test, since rounding those large sums could push the result away from zero.

I'm recalling that JAX on A100-class GPUs uses TF32 for float32 matmuls by default, which could push large sums off exact zero.

I'm checking whether TF32 precision on GPU could push those large sums off exact zero, and the next question is whether the net's own evaluation would show that loss of precision.

I'm checking whether the network's intermediate activations are integer-exact in the first pass, and whether TF32 rounding could affect the result. The inputs are small integers and the weights are exact in TF32, so I'm concluding that this check can't tell the two cases apart.

I'm weighing whether to build a small test net to see if L2 receives rounded values when L1 outputs something like 1000.5 or 4097. The check costs one pass over the 23 positions, and it would tell me whether evaluation is fp32-exact. Multiplicities can't be kept small enough to rely on bf16 or TF32, so the exact approach only works if fp32 is confirmed.

I'm considering a simpler route: instead of exact atari detection, I could make the self-atari exemption apply only when the captured group is provably small and fully explored, using a frontier signal that flags any stones whose liberties haven't been fully traced.

I'm trying to find a way to mark stones far from the group's frontier, so I can tell whether a group has been fully traced within a given radius. Counting walks by propagation doesn't give me an "exactly k steps away" signal, so I'm considering a different seed type that flags distant stones directly.

I'm trying to make the atari check conservative by seeding an "unexplored" signal on each stone and propagating it outward, so that a group only counts as fully traced when the signal dies out within the radius. The catch is that subtracting walk counts to isolate "exactly K steps away" doesn't work with simple propagation.

I'm noticing that walks of exact length k on a grid only reach cells at distance k or less with matching parity, since the grid is bipartite and walks can bounce back along an edge. So a plain walk count can't tell me exactly which distance a stone sits at.

I'm working out whether a two-hop propagation step, with a middle stone gating the signal, can extend the atari check to groups whose safe seeds sit further away than three hops.

I'm looking at whether a two-hop step can be done in one layer, and realizing that the middle-stone gating needs a separate relu stage, so it can't be folded into the same pass. Without gating, a stone at a straight-line distance could leak signal through a non-own middle point, which would wrongly mark a group as traced.

I'm weighing a soft gate that subtracts a large penalty whenever the middle point isn't our own stone, so invalid paths get zeroed out. The catch I'm stuck on is that the penalty also hits paths whose far stone carries no signal at all, so the gate has to be checked against that case.

I'm checking whether a two-hop gated propagation can be built from AND units, one per path, since each path needs its own gate on the middle point.

I'm seeing that a two-hop gated step would need its own relu stage, but the middle-point gate is just the standard one-hop propagation at that point, so each extra hop costs a layer. That means the layer budget caps the reach at about four hops unless the seeds themselves carry longer-range evidence, like straight or bent pairs of stones.

I'm working out how bent-pair seeds could extend the reach of the atari check: a diagonal pair of liberties whose corner cells are both the group's own stones is guaranteed to be distinct, so it can seed the signal near every stone. I'm testing that on the white group from game 34, where the liberties at (0,6) and (2,8) are reached from stones at (1,6) and (2,7) through the bent pair around (1,7).

I'm working out how to seed the L2 layer with pair-based evidence, so that the atari check can reach stones whose nearby liberties are guaranteed distinct, like the bent pair around (1,7) in game 34.

I'm now looking at a seed that counts distinct liberties in the neighbourhood, treating two empty points adjacent to the same locally connected group as the evidence I need.

I'm now defining the liberty count for a local group: empty points that touch either the stone itself or one of its own neighbours, counted without duplicates across the distance-two neighbourhood.

I'm noticing that a straight neighbour's emptiness combined with its link to an own middle stone is a per-pair quantity, so the same units could be shared across different centres rather than built fresh for each one.

I'm working out whether per-stone liberty counts can be corrected for double-counted diagonal points, by subtracting the cases where two own neighbours share the same empty corner.

I'm checking whether a per-stone overcount from diagonal shared liberties can be fixed with a unit for each diagonal corner, where the corner is empty and both middle neighbours are own stones. That would cost four units per point per colour, and I'm wondering whether a 2x2 block shared between neighbouring centres could cover them instead.

I'm checking whether the overcount from diagonal shared liberties can be corrected cheaply, but a unit per diagonal corner works out to too many units per color. I'm wondering if the "multiset liberty count" could be used just for the atari question, since it only needs the liberties adjacent to the move. I'm also considering whether propagation along straight lines could speed up the reach.

I'm considering whether the L1 spare units could carry a per-point indicator, like whether a stone has any empty neighbour, so that the L2 seeds can use bent and straight pairs more effectively.

I'm checking whether straight pairs could seed the atari check more cheaply, but two separate seed units per point per color would push the layer past its unit budget. I'm noticing the COLOR unit isn't used anywhere right now, so I'm wondering whether it could be freed up to make room for the gating carries.

I'm checking whether a single linear seed unit with threshold 3 covers the cases I need, which it does for the simple ones. What it misses is a stone with no liberty of its own that sits beside two neighbours each carrying one liberty, arranged straight or bent around an occupied corner, and that's where I'm stuck on whether any linear term can express it.

I'm checking the white group from game 34 again, where the bent pair at (1,6) and (2,7) around corner (2,6) is own-occupied, so it might still seed the atari check. The group's only liberties are (0,6) and (2,8), and no existing seed fires there, so a corner-occupancy rule could catch it.

I'm noticing that a solid 2x2 block of own stones guarantees distinct liberties for any two of its stones, since the shared neighbours inside the block are occupied. So a block-level unit could count liberties over the four stones and fire once at least two of them are free, without needing a separate corner term for each pair.

I'm checking whether the 2x2 block seed could be built at L1 straight from the raw input, counting the empty points around each block of four own stones. That would avoid the extra layer cost, but the block's external neighbours are the open question.

I'm checking whether a 2x2 block of own stones can carry its own seed at L1, since the block's external neighbours are the only empty points that matter, and a single linear threshold over them could capture the "two stones with libs" case. I'm now turning to straight pairs and whether a vertical line of three own stones gives the same kind of shortcut.

I'm checking the line seed for a vertical run of three own stones: if the two outer stones together have at least two empty neighbours, or the middle stone adds more, the group has two distinct liberties, since those neighbourhoods don't overlap except through the middle stone.

I'm noticing that a straight triple of own stones has all its liberties distinct, so its count is just the sum of empty neighbours around the three stones. That would let one linear threshold serve as the seed, though 126 such triples per colour would overrun the spare units alongside the block seeds. I'm also considering whether adjacent dominoes, whose liberties never overlap, could be handled the same way.

I'm noticing that an L-tromino's liberties can be counted as a set, with the shared corner counted once, so the same linear threshold trick works for any small connected shape of own stones. That would make "all own and at least two liberties" a single unit at the first layer for each such shape.

I'm weighing which small fixed shapes of own stones could serve as seeds, since any connected set whose distinct external liberties are counted stays linearly thresholdable. Dominoes, straight trominoes, L-trominoes and 2x2 squares each cost a different number of units, and I'm checking which gives the best coverage within the layer budget.

I'm now weighing whether to keep the per-stone liberty counts at the first layer and spend the leftover units on 2x2 block seeds, and I'm considering whether seeds built from larger shapes could give each stone more reach at the next layer.

I'm going to look at where the current rule fails on real game positions, but I'm holding off on running my network's activations in the sandbox. Instead I'll study group-structure statistics on the boards directly.

I'm weighing whether to check how often the current rule fails on real game positions, using group-structure statistics rather than running the network itself. The aim is to prioritise fixes for repetition losses and self-atari from false captures, since the net already beats weak opponents but is far from reliable against stronger ones.

I'm going to set the repetition question aside for a moment and think about how to measure where the seed rule actually breaks down on real positions, which is the step I'm about to take.

I'm weighing whether the L6 capture check could carry the atari signal directly, leaving the earlier layers free for the nonlinear conditions that SAVE, SELF and ATT need.

I'm working through how to make the output layer's SAVE, SELF and ATT features into single units at layer 6, using the propagation values already built up through the earlier layers, instead of keeping them as separate AND conditions.

I'm working through whether the big-or-safe-neighbour cases could be handled by a single relu over the neighbours: an own-stone-minus-safety term is only useful if a safe friendly neighbour doesn't cancel an adjacent atari group, and that's exactly the save case I'd want to keep firing.

I'm considering moving the domino seed down to layer one, so that each adjacent pair of own stones checks its own liberties directly rather than waiting for the second layer. That would free up depth for the later hops, though I still need to confirm the single-stone term fits within the unit budget.

I'm working out whether a domino-based seed at layer one could replace the single-stone and pair seeds, which would save a hop. The budget looks too tight, though: per-colour dominoes and singles together would push the first layer past its limit. I'm wondering whether dropping lone stones and relying on dominoes covers enough of the cases.

I'm considering whether the carry channels could be shared between colours, so that safety information for one side doesn't need its own dedicated units, though the risk is that a safe stone's signal bleeds into an adjacent stone of the other colour.

I'm asking whether the ≤2-liberty layer is worth its units: dropping it would free space for better atari detection, though its own-group-with-three-liberties signal is what makes connecting to a safe group count as a save in SELF.

I'm trying to turn the single-stone atari check into a self-atari test: a move is self-atari if its group ends with only one liberty. Approximating an adjacent own group with at least two liberties as contributing one extra liberty works for most cases, but it would miss a two-point eye fill, where the group's remaining liberty is the single empty point left after the move. That eye-space situation is probably something the interior feature has to cover.

I'm weighing whether the ≤2-liberty layer earns its units, and I'm reconsidering the attack signal, which may matter less than the other features. Looking at the game logs, the margins are close, so the bigger issues are likely losing large groups through misjudgment and wasting territory. I'll pull up a lost game, the one ending at −12.5 as black, and follow its progression to find the main mistakes.

