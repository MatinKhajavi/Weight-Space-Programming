My stones are spreading out with no clear shape, scattered across the board including several on the first line. The opponent has built a solid wall on the left while mine stay thin and drift toward the edges. I think the tent-style evaluation is steering me toward contested open areas, and that is why I keep ending up on the edges and on that early first line.

I'm weighing a few concrete changes: penalize first- and second-line moves more heavily in the opening and middle game, favor moves that connect to my own stones or make solid shapes, and reduce the spreading. I also want to consider replacing the tent-based contested-area heuristic with a standard influence-based valuation of expected territory change, since the opponent's solid wall suggests I need to play more compactly.

I'm considering a phase-dependent edge penalty instead of a fixed one, since first-line moves may still be needed to close borders in the endgame. A flat negative prior could push those moves below the pass option, so the penalty would need to scale with how many empty points remain.

I'm considering a phase-dependent edge penalty so that first-line moves get discouraged in the opening but remain available in the endgame, since a flat penalty might push them below the pass option. I'm also weighing how to measure connection and shape, and whether weak groups with few liberties need reinforcing. My suspicion is that the tent kernel is too wide, which makes isolated points far from my stones look most valuable and drives the scattering.

I'm working through a more principled valuation where each stone's influence depends on its strength, not just its position. A move's effect on score would come from how the kernel shifts ownership near it, but a lone stone surrounded by enemies shouldn't count for much. So I'm thinking of weighting each stone by its liberties, giving full influence to chains with at least three.

I'm checking whether weak groups need protection: a chain with few liberties exerts less influence, so the points around it look contested and worth playing. That would push moves toward reinforcing them, which fits the scattering problem. I'm also considering that the opponent's perspective matters, since a move that denies their gain is symmetric to one that gains ours under this linearization.

I'm now checking whether the influence weights should drop for stones with few liberties, and I'm considering a single phase unit that lowers the edge penalty as the board fills. I'm also reconsidering whether the tent kernel's distance-two and distance-three values are too generous, since they seem to draw moves toward isolated points.

I'm looking at the passing pattern now: in game 17, after passing around ply 82, the opponent kept playing inside what I'd been treating as my area, and each time I answered with a move rather than a pass.

I'm considering a structural change to reach across larger chains, where each propagation layer would take two hops using an AND step over the previous layer's results. I'm also noting that the passes in game 20 fall inside our own interior area, so declining those moves seems fine.

I'm working out a two-hop version of the liberty propagation, where each layer checks not just adjacent points but points reached through an intermediate same-colored point. The idea is that a point counts as a witness only if the path through its neighbor is connected, so I'm writing the condition for that second hop.

I'm working out a way to cut the cost of two-hop propagation: rather than tracking every pair of points along a path, I'd store a single gated unit per point that marks whether a same-colored neighbor has a witness. That keeps the unit count manageable, though I'm still checking the negated-OR form for it.

I'm working out whether an extra per-point unit, one that records only a same-colored point's own color plus its neighbors' witnesses, can give two-hop reach in a single layer without needing a unit for every pair of points.

I'm working out whether a two-hop reach can be built into each layer by adding an extra per-point unit that flags a same-colored neighbor with a witness, rather than a unit for every pair of points. The next step would be defining that gated unit and checking how it feeds into the following layer's propagation.

I'm realizing the H unit just reproduces the witness information at the same reach, so stacking it doesn't extend how far a layer can see. Each layer only moves information one hop, since the gating needs an AND over neighbors and the unit can't do that across a chain in one step. So a two-hop unit would have to be built differently, not just as another propagation term.

I'm considering whether a single unit could check a whole straight path of same-colored points at once, but that would still need one unit per path, which is too many. I'm now looking at a per-direction run-OR for each point, where a witness is found anywhere along a straight run, to see if that gets longer reach without the explosion in units.

I'm trying a run-based witness check: for each direction, look along a straight line of same-colored stones and see whether a witness appears at the end of the run. The difficulty is that the run length is itself variable, so a single linear unit can't express "any k up to the run length" without a product term for each k.

I'm testing whether a single weighted sum can cut off a run of witnesses at the first break: giving each position a decreasing weight so that a gap in the run cancels everything beyond it, but witnesses before the gap still count. The first attempt with equal weights fails, since a break near the end doesn't cancel an earlier witness, so I'm trying geometrically decreasing weights instead.

I'm checking whether a weighted sum with geometrically decreasing weights can cut off a run of witnesses at the first break while keeping those before it, and the bounds look like they hold.

I'm checking the weighted-sum construction at the end: with decreasing weights, a witness before the first break keeps a positive total, while anything past the break gets cancelled. The bound on the break penalty comes out to a margin of about 8/9 of the witness's weight, so the sign holds in both cases.

I'm checking whether a single relu unit can turn the run-witness sum into a clean yes/no, since the sum can go negative when no witness is present. The open question is how to fold that sign-flipped term into U's update without letting it add to U.

I'm checking whether one unit per axis could handle both directions at once, but summing a positive run check in one direction with a negative one in the other could cancel a real witness, so that doesn't hold up.

I'm checking whether scaling the penalty term by a large constant could work, but a negative value would push the update above the cell's own state, so I'd need a clipping unit to keep it bounded. I'm also wondering whether it's worth tailoring the cost to opponent chains only, since a large capture would be misjudged for both colors.

I'm weighing whether to keep the straight-line propagation for the opponent's flags alone, since that's where the big false captures come from. I'm also considering a simpler fix: checking a suspected capture with a self-atari test before acting on it, which would cut most false captures without the unit cost.

I'm considering whether a chain-size check could help: only counting a flagged stone if its chain has no other liberty, since a real capture would leave that chain with exactly one. Any other liberty in the chain would point to a false flag.

I'm now seeing that the failure may not be about reach at all: a chain with only two liberties at opposite ends gives no single stone two liberties in its patch, so no witness fires and the whole chain gets flagged as in atari.

I'm considering a different kind of evidence: instead of a single witness, a chain has at least two liberties if two of its liberties fall in different classes of some partition of the empty points, such as checkerboard parity or row and column residues mod 2 or 3.

I'm now thinking about cheaper ways to tell whether a chain has two liberties, since counting them per point gets expensive fast. Partitioning the empty points into classes like checkerboard parity would work if two different liberties always fell in different classes, but that doesn't hold for every pair, so I'm still looking for a partition that catches them.

I'm adding a second evidence source: each chain gets an "has a liberty of parity A" flag and an "has a liberty of parity B" flag, and two of these both set means at least two liberties. That still leaves chains whose liberties all share one parity, so I'm considering a second partition by row and column parity to split them further.

I'm now checking whether two evidence flags, one for liberties of each checkerboard parity, could replace the four-class version, since the four-class flags would cost 648 units. Then I'd combine them with the witness flag so that two liberties of different parity, or a witness, marks the chain as having at least two liberties.

I'm considering swapping the U3 flags for class-based detection: injecting the two-liberty witness into both parity flags, so that "at least two liberties" reduces to both flags being set, which costs the same two units per color as the U2 and U3 pair it replaces.

I'm working out whether the U3 flags can be dropped in favor of two parity flags per color, since that keeps the same unit count. The catch is that the six-flags-per-point budget only leaves room for four flags beyond own and opp, so I can keep either the parity pair or the U2/U3 pair, not both.

I'm asking whether the own/opp flags can be dropped, since each update needs the neighbor's color to gate it. Writing the flags in positive form, a flag is set only when the stone has evidence and is its own color, and the neighbor update becomes an OR over neighbors gated by the cell's color, which still needs the color bit.

I'm checking whether the own/opponent gating can be avoided by encoding the color directly into the flags, so the neighbor update needs no separate color input and fits the unit budget.

I'm weighing whether dropping the own/opponent flags and folding each color into its own parity flags could free up enough room, but I'd still be one flag short of keeping an atari-count flag per color within the budget.

I'm checking whether merging own and opponent into one signed plane would save a unit, but a linear encoding of the sign doesn't give me a clean gate for the neighbor update, so I'd still need two values per point. That leaves the budget at six units per point: own, opponent, and four propagated flags.

I'm considering dropping the U3 flags and using two checkerboard-parity flags per color instead. The catch is that without U3, a neighboring own chain with three or more liberties would count the same as one with exactly two, which could produce false self-atari signals when I fill a point between strong groups.

I'm weighing whether to drop the U3 flag for the opponent and keep class-based detection for its two-liberty chains, since that would fix the false captures. The cost is that a neighboring own chain with three or more liberties would then look the same as one with exactly two, so self-atari checks could misfire.

I'm weighing what to lose if I keep only the two-liberty flags: false own-atari signals are wasteful but less harmful than false captures, while the atari bonus for attacking would be gone. I'm also wondering whether the attack bonus could be approximated from something cheaper, or whether class-based flags for both colors would leave room for a rougher estimate of the three-liberty case.

I'm now considering whether an unfinished idea about a cheaper estimate for the three-liberty case at L6 could work, and then turning to which partition of empty points to use: checkerboard parity, or a three-class scheme with flags.

I'm checking whether a chain has at least two liberties by testing the two checkerboard-parity flags together, with the witness injected into both, so that "two or more" means both flags are set. For the unsafe-check, I'm working out the not-form version of those flags and how it combines when the chain has a liberty.

I'm noticing that a stone's neighbors always sit on the opposite checkerboard parity from the stone itself, so any liberty of a stone is at parity 1 minus the stone's own parity. That means the class evidence for a stone can be read directly from its own parity.

I'm noting that a stone's neighbors always sit on the opposite checkerboard parity, so a stone with any liberty gives evidence for the class opposite its own parity. That makes the chain's two-class check depend only on which parities its liberty-bearing stones have. I'm going to see how far this covers the two-liberty cases before trying a finer partition.

I'm weighing whether to cut the per-point cost by dropping propagation on some of the 81 points, such as letting own chains propagate only in part of the board.

I'm weighing whether dropping the opponent's U3 flag and using parity-class flags for opponent chains would keep the same six units per point. That would sharpen atari detection for opponent chains but lose the atari bonus, and I'm still wondering whether the reach could stay at three steps.

Parity doesn't help in these cases, since the two liberties sit on the same checkerboard parity. Reach is still the real problem: the stone at (7,0) is four steps from (3,0) along the chain. I'm considering straight-line run propagation, since walls along a line could carry the witness across longer distances.

I'm thinking about whether run-based witnesses along a straight line could carry the liberty signal further, but the nested conditions make it nonlinear, so I'm looking at making the run detection depend on liberties directly instead.

I'm working out a straight-line version of the liberty check: along a ray from a stone, the first point that isn't the same color should be empty. I'm sketching a linear form that sums empty indicators along the ray and subtracts opponent-or-edge terms, so the detection stays within the unit budget.

I'm checking the ray-based idea: for each direction, a weighted sum of empty and opponent indicators along the ray, with decaying weights, so the sign tells whether the first non-same-color point is empty. The catch is the cost, since that is about 648 units per point at the first layer, which blows the budget.

I'm weighing whether counting rays from a stone that reach an empty point could give a lower bound on liberties without overcounting. Each direction's ray either finds an empty point through same-colored stones or it doesn't, so two successful rays would mean two distinct liberties. The cost is four units per point per color, which together with the own/opponent flags would use up the 512-unit budget for one color's rays alone.

I'm now checking whether the ray witnesses could be normalized so that a positive value reliably means a ray found an empty point, since the raw values shrink geometrically with distance. The concern is the threshold: the smallest positive value at the far end is tiny enough that a fixed cutoff would be hard to separate from zero, and the unit count for own and opponent rays together still looks tight against the budget.

I'm weighing a ray-based witness with decaying weights, where the sign of the weighted sum tells whether a straight line from the stone hits an empty point before any opponent stone. The decay has to be steep enough that the first non-same-color point outweighs everything beyond it, and the smallest positive values this produces are tiny, so the thresholding is the delicate part.

I'm working through whether ray-based witnesses could give a cheap lower bound on liberties, but the normalization needed to turn tiny positive ray values into clean binary signals adds layers that eat the budget. I'm now wondering whether precomputing chain-segment connectivity at the first layer could make propagation reach further.

I'm trying to find liberties through a run of same-colored stones along a ray. A liberty can be the first empty point straight ahead, or an empty point beside a run stone, and I'm looking at a weighted sum that marks whether such a point exists.

I'm adding a perpendicular term to the ray sum, counting empties beside the run only while it stays unbroken. The concern is that those side terms past the break could drown out an opponent break, so they'd need to be capped at half the break's weight.

I'm making the perpendicular side-liberty terms small enough that they can't override an opponent break along the ray. With weights decaying faster, like 5^-k, the total contribution past the break stays below the break's own weight.

I'm checking the perpendicular terms at the break: a liberty beside a stone just before the break carries weight a_j, which is enough to outweigh the opponent break's -a_j plus the later terms (at most 1.25a_j in the worst case).

I'm checking the perpendicular terms at the break again with a different scale: sides before the break get a weight of at least twice the break's own weight, while the opponent break plus the later terms total at most 1.2 times it, so a valid side liberty still wins.

I'm checking that a ray's signal stays positive only when it finds a real liberty: the perpendicular terms are scaled to at most a fraction of the opponent break's weight, so a valid liberty still wins. That gives one unit per stone and direction, covering a cross-shaped sub-chain through the stone. The open question is whether those cross-shaped counts can show two distinct liberties rather than just one.

I'm counting the cost of the diagonal overlap: a point like s+d+e can be a liberty of both the +d run and the +e run, so summing per-direction witnesses would overcount. Four units per point per color comes to 648 across both colors, which won't fit in one layer, so I'm starting to think this ray approach is overengineered.

I'm now looking at a simpler fix for the repetition losses: the capture term in the SA formula is what makes this corner move look attractive, since it adds +2 even when the stone's chain would only have one liberty. Dropping the capture bonus would fix the false positive but would also penalize legitimate captures that happen to look self-atari, so that tradeoff is what I still need to weigh.

I'm thinking about whether the capture bonus should scale with how reliable a flagged chain is, since large chains flagged by the heuristic may be false. I'm weighing a size proxy based on flagged opponent stones within radius 2, though genuine large-group ataris also happen against a weak opponent. Next I want to check real game records to measure how often captures are false versus true.

I'm going to check the game records with a rule-based script: for each of the network's moves, I'll see whether a true capture was available, whether the move actually captured, and whether it was self-atari. That should show how often the capture bonus led it wrong.

