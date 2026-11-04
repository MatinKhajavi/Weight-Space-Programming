Checking the captures: about 28% of them (922 of 3265) have 1 or fewer other liberties. Blocking captures that leave the stone with few liberties would cut off a lot of them, so that's worth weighing.

I'm checking how many captures would be cut off if I blocked the ones that leave few liberties, which turns out to be a large share. That makes blunt blocking too costly, so I want a more precise way to separate the false ataris from the real ones.

I'm weighing whether to block only the captures with two or more opponent stones nearby, since a real big capture rarely shows up with few liberties. Still, the cost of cutting off real captures looks too high, and I'm considering whether reach can grow if the tent computation moves to an earlier layer, though that wouldn't free any units.

I'm considering whether the strategic tent computation could be cheaper: maybe a per-point value using a clipped ownership measure instead of the full 243 tent units. I'm also weighing whether dropping the own-chain flags would free enough budget, though that would weaken the escape and self-atari detection.

I'm considering a different strategic evaluation that stays linear in own and opponent stones, with a few nonlinear terms. Another option is to accept the reach limit but penalize self-atari captures of big chains, adjusting the SA unit so it discourages those moves.

I'm testing whether a capture-bonus penalty can stay net-negative for big self-ataris: the adjacent flagged stones and the wider flagged count together might separate a large chain from a small one.

I'm testing a smaller weight on the dist-2 flagged count, since at 1.5 a real two-stone capture into a hole still gets blocked. With 0.5 the penalty on a small capture drops, but I need to check whether a big self-atari still gets penalized enough.

I'm weighing a mix of options here, since 1.5 gives a marginal result: the penalty on that small capture (about 12.5) barely undercuts the 14-point bonus. The flag count over the second ring is a crude proxy for chain size. A better test might be how often a large flagged chain next to a self-atari capture turns out to be a real capture versus a false one, and I'd need the data to answer that.

I'm weighing which failure is worse: blocking a real big capture, or allowing a false capture that gets punished. A repetition loss would end the whole game, so avoiding cycles is the priority. The cycle pattern is the one to pin down: we self-atari, they recapture two, we retake one, and the position repeats.

I'm considering dropping the own-chain flags and approximating support locally, checking whether a neighboring own stone has another liberty, since that would free budget for reach at the cost of escape and self-atari detection.

Still working out how to fit the strategic evaluation into the layer budget, since each layer already spends six units per point on own and opponent stones and the four flags.

I'm working out whether a positive-form encoding could recover the color of a stone, since the negative-form flags leave c(s) ambiguous when both classes lack evidence. Either way the gating still needs "same color as s," so merging own and opponent into one indicator doesn't obviously help.

I'm now asking whether flags could be propagated through empty points as well, to capture region connectivity, but that doesn't look useful. Since every layer already spends its free units on the own and opponent indicators, the question becomes whether the per-point flags need to cover all 81 points or only the ones that matter.

I'm weighing whether a horizontal or vertical run of same-colored stones through s could carry long-range evidence, since a straight line's ends and side neighbors would reveal liberties without spending a propagation layer on each point.

I'm weighing a cheaper way to handle the disastrous self-atari cases, like penalizing those moves more when the capture looks uncertain. An ensemble with the old T-star system is out, since it misses big chains most of the time. A better candidate is the exact star-based evidence, which avoided most false atari flags for big chains, though it would need corner corrections and extra budget per layer.

I'm checking whether the star-based evidence can be approximated more cheaply: exact star counts help a lot, but they need per-point, per-axis units that won't fit in the 25 free units at L1. Adding the straight-triple case captures most of the gain, though it's still too costly at this layer.

I'm trying to fold the straight-triple evidence into the L1 budget by dropping the end-stone term and using the L1 liberty counts directly, but that leaves 324 units at L1 and pushes L2 to 486, which leaves no room for the NP flags there.

I'm checking whether a three-stone straight-line pattern can supply the evidence I'm missing, though it would cost a layer of reach. Comparing that against the current option, I'll run my evaluator on both setups, including the star-based evidence, to see which gives fewer false captures.

I'm now checking the reach-4 design A' at L2: its evidence comes from the stone and its neighbours, and each further layer adds one step of distance. Then I'm sketching design B, which spends the L2 budget on straight-triple terms instead of the NP flags and so trades reach for better evidence.

I'm now checking a variant where the straight-triple evidence is computed directly from the input at L1, requiring all three stones to share a color, which would cover the case where the middle point has no liberties. That gets the cost down compared with routing it through a later layer, but at 324 units for the per-axis, per-color terms, L1 would come to 810 units, which is well over budget.

I'm checking whether merging the availability and two-liberty evidence into one unit still works, since the L2 neighbour terms would need a nested relu that the layer can't compute inline.

I'm checking whether tri3 can be restricted to just the opponent side, which would bring the L1 total down to 648 units, though I still need to confirm that the evidence it covers is worth the cost.

I'm asking whether the own/opponent pass-through at L1 can be avoided by folding the gating directly into one unit that combines c(s) with the evidence terms. The evidence terms won't fit that way, so I'm checking whether the layer budget still works out.

I'm now checking the alternatives against each other: keeping parity and the single-stone evidence at reach 4, versus adding the straight-triple terms at reach 3, with the star-based evidence as a third option. If restructuring the triple terms gives a clearly better result, it would be worth the reduced reach, but I still need to work out how they'd fit.

I'm checking whether moving the triple-stone evidence into layer 2 can work, since the parity evidence would then need to be carried forward through that layer, which the budget doesn't allow.

I'm trying to combine the two class-seen bits per color into one unit, but a count can't tell which class was seen, and propagating an OR isn't linear, so that doesn't work directly. I'm considering whether a self-sufficient evidence type like the e1, triple, or star terms could replace the separate class flags.

I'm wondering whether parity was actually what helped the third pass catch captures, or whether the gain came mostly from dropping the T-star overcount, which was wrong for big chains most of the time. If the latter, exact local evidence might work without parity, so one flag per color could cover the two-liberty case and leave room for more. I'll set that against the parity-based version to see which one holds up.

I'm now weighing a cheaper approximation: exact star evidence would be ideal, but I can try using e1, adjacent-pair, and tri evidence together instead, which avoids the perpendicular-pair overcount and might still fit the budget.

I'm checking whether the overcount in the star-style sum only shows up when the stone has no liberties of its own, since a stone with a liberty or two already gives a genuine pair with any liberty-bearing neighbour.

I'm working out whether the overcount can be avoided: when a stone has no liberties of its own, the neighbouring stones' liberty counts double up on the shared corner, so the naive sum is wrong. I'm considering an exact local condition that handles the zero-liberty case separately, though a nonlinear indicator for "no liberties" doesn't fit the layer's linear form.

I'm weighing whether the overcount in the star-style sum only matters when a stone has no liberties of its own, since then neighbouring liberty counts can double up on a shared corner. I'm considering whether a cheap approximation that drops the perpendicular pairs could still fit the budget.

I'm checking whether the overcount only bites when the stone itself has no liberties. The L-shaped case where a single liberty gets counted twice is common in big chains wrapping their last liberty, so I'll gate the sum on whether s is lib-bearing and see whether that keeps the evidence exact enough.

I'm checking whether the straight-triple evidence is worth its cost at this layer. I'd need to see what it catches that parity plus single-stone evidence misses, specifically chains whose only multi-liberty evidence comes from a straight line of three with no liberties on the middle stone.

I'm comparing three options numerically: parity plus single-stone evidence at reach 4, adding straight triples at reach 4 or 3, and using star evidence instead, and I'll keep the simpler reach-4 setup unless the reach-3 triple version clearly beats it.

