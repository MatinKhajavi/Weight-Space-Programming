Right now the territory count looks worse for us: White has roughly 45 points to our 35, with White holding the left and bottom in game 0. Our filling moves inside our own area don't lose points, but in the middlegame each wasted tempo costs us.

I also need to improve the positional evaluation itself, so I'm weighing an influence-based way to score moves within a fixed-depth search.

I'm working out how to score a move's territorial gain from influence values: contested points, meaning those with small absolute influence, count most, while points already clearly ours or theirs count little. I think a two-stage calculation could fit in the network, with one layer computing a per-point contestedness value and the output layer summing those linearly into the move logits.

I'm working through a tent-shaped contestedness function for each point, which needs three units per point, so about 243 at that layer. I'm also considering a cheaper one-sided version that values a point only by how much it can still flip toward us.

I'm weighing a ramp-shaped contestedness function instead of a tent: one-sided, with two units per point, where points the opponent dominates are valued at a constant cap. That's less of a fit for contested points specifically, but it still scores "not yet ours" territory, and I'm questioning whether playing into opponent-controlled areas is worth anything against a solid opponent.

I'm sizing up how much room the network has left: the per-point influence layer would need roughly 160 to 240 more units, and the current budget is already packed. I'm considering whether some existing feature, like the liberty or eye blocks or the depth of the own-side propagation, could be trimmed or computed another way to make space.

Dropping the separate opponent parity blocks would free a block in the earlier layers, but the L6 layer is the real constraint, since it's the only one feeding the logits directly. Getting room for the tent units there means cutting roughly three blocks' worth of existing features.

Now I'm weighing which of the remaining L6 features could be merged to free up room, looking at the eye and own-territory blocks and whether the liberty block can fold into the contestedness units. I'm also considering whether self-atari and atari-capture can be combined.

Now I'm weighing whether the eye feature can be folded into the self-atari term, since an eye point with all-own neighbors that have enough liberties should get no penalty.

I'm checking whether the eye feature can be absorbed into the self-atari term: it would need a bonus that switches on only when every neighbor is our own, and a plain sum of own-neighbors doesn't do that without a nonlinearity.

Still checking whether a two-unit version could work: an opponent-leaning and own-leaning split per point, where the absolute influence is their sum and the contestedness is roughly a constant minus that sum. The catch is that this goes negative for strongly decided points, so it penalizes moves there rather than just ignoring them.

I'm now considering a cheaper version that uses two units per point, one for own-leaning and one for opponent-leaning influence, so that their sum gives the absolute influence. Penalizing that sum would favor balanced points between stones and disfavor both deep own and deep opponent areas, which seems like a useful signal. I'm checking whether it would pick out good moves, like a point in the middle between our stone and theirs, and not just the points right beside our own stones.

I'm working out whether to restrict the influence sum to empty points, since stones already give large influence values that would penalize moves in crowded areas for the wrong reason. One option is to mask occupied points inside the contestedness term rather than dropping them outright.

I'm checking how to handle occupied points: a masked occupied point gets zero in both halves, so it reads as perfectly contested, which isn't what I want. Edge points also see fewer neighbors and so get smaller penalty sums, which would bias the logits toward the edge. I'm considering normalizing the kernel per point so each move is scored by its average absolute influence over the points it reaches.

I'm now looking at whether the liberty-style contact bonus can be expressed through the same influence terms, since it seems to depend on opponent stones adjacent to the move.

I'm considering whether the influence sums can be computed without masking occupied points, using a split into positive and negative parts so that a linear function of the influence values can still encode raw stone colors near the candidate point.

I'm checking whether the liberty-style contact bonus can be approximated by a sum of influence values over the neighbors of a point, which would collapse the separate contact feature into the same kernel. Dropping the opponent-territory term and replacing it with an own-leaning penalty would bring the layer to about 486 units, and I'm starting to define the influence kernel more carefully.

I'm weighing the logit as a sum over nearby influence values, with own-leaning and opponent-leaning terms weighted differently, and wondering whether the contact bonus can ride along in the same terms.

I'm weighing whether a clipped tent over the influence magnitude is worth the extra unit, since the unclipped version penalizes far-decided points too harshly. The current two-unit split can't express the clipping on its own, so I'm considering adding a third unit to get it.

I'm weighing a four-unit version that would give the exact tent shape, but at 324 units it's too costly. A two-unit approximation using c(I) = −|I| looks more practical, though I need to check whether the magnitudes stay sensible, since neighbors of own stones can reach values of 4 or more.

Now I'm settling on the plan: the tent-style unit is the real cost, so I'll use a two-unit approximation with a linear penalty on absolute influence, and drop the separate line-contact feature. Early on all influence is zero, so the line prior and tengen will decide, and the middle points only matter after the first stones go down. I'm also wondering whether the approach will make the play too passive against an RL-trained opponent, though the tactical features should still handle the fights. I'll try it quickly by reverting the epsilon changes and swapping LIN and OT for the W and O units, using kernel weights of 2, 1, and 0 at distances 0, 1, and 2.

Now I'm sorting out the contact term: with the coefficient at about −0.2 on the summed influence of neighboring points, opponent-adjacent points come out positive and own-adjacent ones negative, matching the old line-contact feature. Next I'm checking whether the magnitudes hold up, since a point deep in our area with influence around 3–5 nearby would take roughly −2 from the own-leaning penalty, while contested points stay near zero, and opponent-leaning areas land somewhere around −0.3 to −0.9.

I'm checking the remaining constants: the line values stay as they are, and I'd keep them unchanged. I'm also planning how the own and opponent influence units get filled in, taking the positive and negative parts of each point's influence value separately.

