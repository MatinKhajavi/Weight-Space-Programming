I'm going through the game position: the black stones are losing the bottom and left side to a tightly connected white group, and my earlier moves got scattered. I'm tracing how white's wall in the middle-left formed and where it cut through around the bottom-left, and I'm looking at the stones around move 46 near the edge.

I'm weighing whether a simpler, more robust approach would work better: solid shape that stays connected and avoids cuts, responding to contact by extending. I'm also considering a territory-style heuristic that scores moves by influence, approximated with a contested-points feature.

I'm weighing whether to build an influence-based term for the strategy, scoring each point by the difference between my influence and the opponent's and using leftover capacity in the network for it. Empty areas that nobody claims still need a separate value, which would take another set of units.

I'm checking whether a sum of the contested-influence values can work as the move score. That would reward boundary points where influence is shared, and it would also add a linear term that favors points far from existing stones early on.

I'm comparing the tactical variants so far: v5b at 48%, v5 at 51%, v3b at 57% but with more repetition losses, v2 at 53%, and v4 at 38%. Apart from v4, the differences look like noise, so I'll settle the tactical version before running any strategic experiments.

I'm comparing the old SELF variant, which used a weaker neighbor check, against the newer one that counts own stones in a 3-point radius, and noting that the newer one passes far more often. I suspect that blocking extra moves is costing points, and the big-group liberty count may be stopping us from filling dame or answering contact.

I'm thinking about whether big own groups should be treated as having at least three liberties, since the current liberty count likely misjudges them as having two or fewer. That would stop the rule from blocking moves that fill dame or connect near large groups, but I'd only apply it to groups above a size threshold so truly weak groups still get judged precisely.

I'm defining the per-stone liberty count so that own stones count as having three liberties when they're part of a big group, and I'm working out how to express the "small group, no seed" indicator with a clamped threshold so single stones don't go negative.

I'm working out whether the small-group liberty correction can be made graded enough to avoid the full penalty on near-empty groups, scaling the mass so that single stones and small groups land on a similar value.

I'm working through how a graded mass value could separate small own groups from big ones, but the raw numbers I'm plugging in suggest the weights mostly measure local degree rather than group size. I'm checking where the cutoff at 12 falls, so that a graded penalty on small groups stays meaningful without going negative for single stones.

I'm considering a different way to express "big group" as a non-negative signal, so that own stones in large groups get treated as safe in the self-atari check, but I'm noticing this could let the attack check fire on big groups without any opponent contact, which would be wrong. I'm looking at whether the attack term can rely on the liberty count alone instead.

I'm proposing a per-stone safety unit that is zero for small own groups without a seed and grows with group mass, so big groups get treated as safe in the self-atari check without any opponent contact being needed.

I'm working out how the attack term should behave now that the safety signal is unbounded: the self-atari penalty stays in SELF, and ATT can drop the group-liberty term and rely on opponent contact plus the immediate liberty count.

I'm laying out the layer-by-layer channel layout: the seed and mass channels at each level, with the final level's channels summing to four. Then I'm working out how the self-safety term should sum over the safe-mass and attack channels in the SELF formula, and noting that the safe-mass signal is large in value.

I'm going to build v6 on top of v5b by adding a per-stone SAFE3 term, which is unbounded and zero for small own groups without a seed. I expect the risk that big own groups with exactly two liberties get treated as safe, and the chance that a big atari group gets extended into a dead end, to be acceptable trade-offs.

