For moves at ne=0 with weak own neighbors, the false cases are almost all next to 10+ chains (154), while real captures there are only 80, so adjacency to a big opposing chain looks like a red flag. Small chains look fine, with 151 real captures against 13 false ones. That points toward a size-aware rule: penalize when the adjacent opposing chain is 10 or more, allow it when smaller. The ε-mass idea could serve as a rough size measure, rather than downweighting AP across the board.

I'm weighing a fix that uses local opponent stone density around each point as a proxy for chain size, since the ε-mass channel can't be separated from real evidence. Adding a term that raises the penalty where opposing stones cluster densely could target large chains, though a small gain might not be enough to matter.

I'm sketching a v9 layout for the layer-6 units: keeping the escape and capture units, swapping the own-neighbor check for a relu that also weighs a third term, and wondering whether the single-stone boost (SING) still earns its slot in the budget.

I'm working out a penalty for the no-own-neighbor case: the U1 term gives 2 when there are no own neighbors, U2 gives 1, and U3 depends on the density D, so I can't make the total cancel exactly for every D.

I'm trying to make the no-own-neighbor case cancel exactly, so I'm looking at a correction term that's zero whenever there's at least one own neighbor and fires only in the all-opponent case.

I'm trying to make U3 fire only when a point has at least one own neighbor, so I'm considering counting own neighbors directly in the relu. The ne-gate breaks when the term grows with n_own, so this needs a different structure.

I'm trying a U3 that fires only when there's at least one own neighbor, using a bound of n_own − 4ne + γ(D − t) − 0.5 so that the no-own-neighbor case stays at zero.

I'm stuck weighing a density-based penalty for U3, since density is only a weak proxy for chain size and the gain may be too small to matter. With about 38 passes and 74 runs left, and v4 and v6 still the best configs, I'm inclined to stop tuning this unit and focus the remaining budget on what has already shown results.

I'm narrowing the repetition fix to a precise local pattern: a point with no empty neighbors beyond its own liberties, adjacent to exactly one own stone that itself has only two liberties, one of them the point itself. That would be a per-stone unit computed at layer 6 from the layer-5 carry, so it targets the self-atari cycle without touching general capture behavior.

Still working out a per-point unit for the self-atari pattern, where a point touches a single own stone with exactly two liberties and no own neighbors.

I'm trying to detect a point with no empty neighbors that touches an own stone which has no other own neighbors, using a sum over the neighbor's own neighbors to exclude other friendly support, and checking whether that stone has exactly two liberties.

I'm stuck on a per-point unit for the self-atari case: it needs a conjunction over which of p's neighbors is the lone own stone, and a single relu can't pick out one neighbor without a separate unit for each of the four, which would blow the budget. I'm leaning toward accepting some repetition loss rather than spending the remaining passes on it.

I'm checking whether the ε-mass channel explains why the repetition cases survive, looking at the corner pattern where the stone at (2,1) sits next to two opposing stones and may be in a thinner area than I assumed.

I'm checking whether the ε-mass term actually suppressed the captures at the blob edge, since the estimate says it should have. Pass 9 still gave 0.9993 for (2,0) in game 12, which doesn't fit that. I'm going back through the layer-2 through layer-5 biases and weights to see whether the opponent-side ε term was wired in correctly.

Checking whether the (2,0) move in game 12 came from something other than the adjacency term: the own stone at (1,0) has two liberties, so it isn't in atari, and the self-atari bonus would only give about −10. Working out what else would have to push that move to 0.9993 probability.

I'm weighing whether to keep tuning the self-atari penalty with a per-stone unit that can't isolate the lone own neighbor cleanly, or to return to the v6-style configuration and test it on larger samples. The ε-mass term apparently didn't remove the repetition cases, so I'm inclined toward the second option.

I'm considering whether to cap the per-point penalty so it counts a chain once instead of once per adjacent stone. That would stop the corner repetition pattern, which always has two same-chain stones next to the point, from doubling up, though it would also block some real captures with a weak own neighbor unless the single-stone boost covers them.

I'm weighing whether chain size alone can separate the real multi-stone captures from the false ones, since the counts look very different for small versus large chains. The pattern suggests the real captures are the ones that save our own weak stones, so suppressing them would cost more than the repetition losses. I'm still trying to find a signal that tells a large chain apart from a small one.

I'm trying to make the ε-mass term behave as a graded signal: if ε is small enough, the mass stays below 1 for most chains and only grows for very large ones, so the readout would separate them. The worry is that the corner repetition pattern sits in a region where that mass estimate may not hold, so I'm weighing whether a second, stricter readout unit per stone would separate certain from uncertain captures better.

Next I want to see the actual v5 activations on a corner-pattern position, using the pass-12 test boards, to check how large the ε-mass gets for those chains. Then I'll decide whether to rebuild v5 and check it, or go straight to a v9 design with the ε-mass term, a cap of 16, and the SA units.

I'll check the v5 activations on the testpos3 boards next to calibrate the ε-mass, then re-save v5 first.

