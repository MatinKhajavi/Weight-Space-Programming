I'm checking how the error rate changes with reach: at reach 4 there are about 2,700 false atari flags, and at reach 8 only about 1,000, so reach matters a lot. I'm also working out how far the L2 evidence extends for each level, which gives reach 4 for the case I'm looking at.

I'm looking for a way to get more reach out of the propagation. One idea is that layer 6 can't propagate at all, and speeding up propagation with two hops per layer doesn't work either. I'm also considering whether to run the tent features in a parallel lane, though that would still take up layer 6.

I'm testing whether the straight-run term can be made non-negative, so that it only fires when evidence actually exists along the ray. Letting it go negative and then clipping with a second relu doesn't fix the problem, and letting U' exceed 1 would turn negative flags into anti-evidence for neighbors in the next layer.

I'm weighing whether the straight-run evidence could be gathered directly at layer 1 from the raw input, so it doesn't have to be propagated along chains. The catch is that the perpendicular neighbors and the break point along each ray would need to be checked together, and I haven't settled how to combine them without the sign problem coming back.

I'm weighing whether the ray evidence could be computed per parity class and color, but that comes to 1,296 units, which is far too many. I'm starting to ask whether the exact flags are even needed, since the costly failures are false ataris inside big chains that lead to wrong captures, escapes, and SA miscounts. I'm looking at whether a simple rule could block the repetition disasters that come from SA when a capture is being counted.

I'm considering a heuristic for the SA move rule: treat a capture as uncertain when the nearby opponent stones are numerous, since flags are only reliable for small chains. I'm also weighing whether reducing the tent features to two per point could free enough layer-6 units for a separate no-capture SA unit.

I'm testing whether a density term inside the SA unit can suppress false captures in dense regions. It would reduce the capture credit when many opponent stones are nearby, but it would also suppress real captures of large chains, so I still need a way to keep the capture bonus for those.

I'm checking whether the density term can scale only the capture credit without also pushing ordinary non-capture moves into the self-atari penalty. A density term inside the SA unit would wrongly flag moves with two liberties and no captures whenever the neighborhood is crowded, so it has to be tied to the flags alone.

I'm settling on a different angle: skip crediting captures inside the self-atari unit and give capture a separate, larger bonus instead. That keeps real small captures into holes worth playing, since the bonus can outweigh the penalty, while leaving the false big-chain captures to be handled elsewhere.

I'm working out how to cap the capture bonus so it doesn't grow with chain size, using min(1, Σflags) built from two relu units, which costs about 81 more units at layer 6. With a flat bonus and a large self-atari penalty, the false big-chain captures would net negative and stop being played.

I'm checking whether counting real captures that leave the capturing stone with no other liberties, using the no-capture self-atari measure, would show how often a flat capture bonus would wrongly suppress genuine captures.

I'm checking how many captures would be self-atari if you ignore the captured stones, by playing each move without removing them and counting the new chain's liberties. If that count is one or fewer, the capture falls into the no-capture self-atari case, and I want to see how many real captures that catches, grouped by size.

