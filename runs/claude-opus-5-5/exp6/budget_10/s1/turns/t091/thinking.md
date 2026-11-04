I'm checking how many empty cells lie within Manhattan radius 2 of the position, excluding the position itself. The rule "keep the eye penalty when that count is at most 1" catches all 42 catastrophic cases, but it also discards 81 of the 425 legitimate escapes.

I'm checking whether keeping the eye penalty whenever there are at most one empty cell within radius 2 is worth it. It blocks all 42 catastrophic cases at the cost of 81 legitimate escapes, and I think that trade is acceptable. Next I'm working out a formula for the eye unit that gives full penalty when the empty count is low, even with the gate on, and none when it's high.

I'm testing whether the e2 term can be made one-sided, since a plain linear version would zero out the eye penalty for real eyes in open territory. Wrapping it in another relu won't work, so I'm looking at whether scaling the gating amount by the empty count could do the job instead.

I'm trying a version where the eye penalty is a single linear term with a constant that is large enough to protect real eyes in open territory. Setting the constant to the maximum possible empty count within radius 2 keeps the penalty from dropping to zero for eyes with lots of empty space around them.

I'm settling on a constant κ between 11 and 12 so that the gated term drops to zero once two or more empty cells are nearby, while still giving a positive penalty when there are one or fewer. Taking μ = 1 gives κ = 11.5, which yields 0.5 at one empty cell and 1.5 at none, and clamps to zero at two.

I'm checking the unit with κ = 11.5: when the opponent-neighbour fraction G is partial, say 0.33, the gated term stays positive for moderate empty counts, so the eye penalty still applies.

I'm checking whether partial gating breaks the scheme: for non-isolated points with a fractional G like 0.67, the eye penalty comes out stronger, and it still applies to real eyes that border opponent stones from several neighbours.

I'm working out the eye term with a constant of 12 so the e2 part never drives the penalty negative for real eyes, and I'm noting that a real eye whose partial opponent contributions sum to G ≥ 1 already loses its penalty in the earlier version. The new formula only adds protection when e2 ≤ 1, so I'm carrying the B·occ gating over unchanged and starting to write out the e2 term.

I'm working out the eye term's constant per point: each point gets κ equal to its radius-2 size minus half, so that with one empty neighbour or fewer the gated penalty stays positive, and with G at 1 it comes out to 1.5 minus the empty count.

I'm checking the eye term's constant against the new one-sided design: the constant should be the radius-2 size minus one half, so that the gated penalty stays positive with one empty neighbour or fewer and reaches zero at two. The non-gated case stays at a penalty of at least one, and the partial-gating case still gives a sizeable penalty, so the scheme holds. I'm also noting that the eye magnitude can reach about 13, which is large but fine numerically, and that the output weight scales it further.

I'm setting up the eye unit's weights: the neighbour-ownership and opponent terms stay as they were, but the opponent-neighbour term now gets scaled by a per-point constant, and I'm starting to add an e2 term that counts occupied points in radius 2.

I'm now checking the occupancy term: the positive contribution from occupied points in radius 2 applies to every point, not just eyes, so I need to watch how it interacts with the bias and the gated G term.

I'm adding a penalty term for non-eye points: each own neighbour gets weight 1 + B, and the bias picks up −B·deg, so any point with a non-own neighbour gets pushed well below zero, while the occupancy count from radius 2 only ever pushes eye points upward.

