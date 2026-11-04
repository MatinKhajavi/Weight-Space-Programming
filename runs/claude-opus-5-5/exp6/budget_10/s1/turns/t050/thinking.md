I'm checking whether the 'l3' rule would have caught 49 of the 83 SA moves, which is better than the 'loc' rule. I'm setting S to 1 plus a bonus when l(q) is at least 3, and working out how to keep its values to 0, 1, and 2 with a single relu-based unit.

I'm weighing a threshold on l(q) that would give values near 1 and 2, and noting that a plain linear form can't match the step at 3. Two units or a hedged affine fit are the options, and I'm leaning toward a half-slope version that accepts a partial penalty for the single-neighbor case.

I'm working out an extra term for the SA unit: a local unit at layer 5 that adds one when a own chain has three or more liberties, with the surrounding terms summing over the neighbors.

I'm settling on the extra term: a local unit at layer 5 that gives 1 or 2 when a own chain has three or four liberties, and 0 otherwise. Since a chain in atari has at most one liberty, that term is automatically zero for unsafe points, so it doesn't need a safety gate.

I'm checking the sign of the SA unit's terms: the own-stone and opponent-liberty contributions net out to zero, while the extra term and the safety-related terms enter with negative weights, and the bias ends up as 2 minus the degree.

I'm checking whether pass decisions are costing points in close games. Mean margin is only +4.3, so endgame dame filling could matter. I'll look at whether any contested or neutral points were still open when we first passed, and count the dame once the game ends.

I'm checking the pass decisions and the dame count.

