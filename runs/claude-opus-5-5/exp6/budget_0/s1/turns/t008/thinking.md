Checking the formula against the full-orthogonal case confirms it matches, and now I'm testing it against the corner case to verify consistency there too.

Going back through the logic once more to check for subtle flaws: the bias calculation for (4,4) checks out, IXP's contribution is correctly included, and the near-eye exclusion rule for false-eye points behaves correctly whether on the edge or with diagonal opponents present. I'm also confirming that filling a point in the middle of a solid group is correctly avoided, and now I want to move on to verifying the pass and endgame logic.

Against fixed or uniform-random opponents I expect near-100% win rates since there's no adaptive exploitation to worry about. I'm now mentally simulating the opening on an empty board, working out the logits for center, d3, d2, d1, and d0 points using the bias formula to see how the shape preference plays out early.

Finishing d0's score at −0.1, I compute the softmax over all distances: center gets about 14%, d3 34%, d2 41%, d1 9%, d0 1% — a distribution that seems reasonable. Now I'm considering White's response, likely placing a stone near Black's center move and evaluating the resulting board points.

Working through the numbers, captures clearly dominate the move selection since they land around 14-16 compared to baseline moves at 4-6, making them nearly always the preferred choice. Atari on a lone stone adds enough weight to often but not always win out over other options, and capturing a stone already in atari gets an extra boost. I'm now checking the ladder-escape case where the stone count is at least two.

For ladder escapes, the typical pattern has the extending group left with just two liberties, which can mean the opponent ataris repeatedly until the group dies at the board edge — though weaker opponents might abandon the chase, and detecting genuine ladder-breakers is tricky. This makes me think I should discount escape value when the liberty count is exactly two, since that likely signals a ladder, while escapes with three or more liberties are more robust and deserve full credit.

I also check that captures take priority even when a scoring move looks tempting, confirm that legal ko captures are handled correctly even if the opponent retakes later after a threat, and note that when multiple opponent stones are in atari I should favor capturing the largest group. Komi considerations for Black don't seem relevant here, so I move on to reexamining the reach flags at the fifth level.

If that liberty point would make our move illegal due to ko, it should be masked instead.

With the core design feeling solid, I'm now thinking about using remaining capacity for a global "game phase" feature — tracking total stones on board, carried through the layers, to modulate things like throw-in behavior or InfOwn penalties later in the game.

I'm skeptical some of these ideas are worth the complexity though — phase-dependent pass logits or throw-in boosts don't seem necessary. Let me focus instead on refining InfOwn, the own-area influence penalty based on radius-3 distance, checking how it behaves for territory points behind a wall versus points adjacent to a lone stone in the open — there's a subtlety about how it scores points immediately next to our own stones at the very start.

I should also think through dame-filling at the endgame under area scoring: each dame filled adds a point, so the network should prioritize filling dame before throw-ins, which means dame points with neutral influence and low move count need the right combination of weights from n, territory, empty-point bonus, and the ring term to come out ahead.

But there's a subtlety: if a dame point sits between our group and the opponent's, and filling it reduces our group's liberties to a dangerous count, that could trigger self-atari and let the opponent capture — so I need to account for liberty-count safety before treating all dame as equally good to fill.

I'm trying to add a moderate penalty for zero-dame moves with no capture to keep them ranked above pass but below normal moves, roughly at throw-in level—working out how to express that with an available spare unit given the existing formula structure.

That approach breaks down for n=0, t=2 since the formula goes negative with no penalty, so it's not monotone-friendly. I'm weighing whether to drop CapSize to free up a unit for Z0, considering which matters more: avoiding self-atari by filling n=0 points without capture/save, versus preferring bigger captures when multiple options exist.

Since multiple simultaneous captures and capture-vs-save conflicts are rare while n=0 dame situations are common in the endgame, I'll replace CapSize with Z0 using a relu formula and weight -2.0, while also considering how this interacts with moves that merge groups or reduce own liberties in subtler ways.

So I'll finalize updating layer 6 by swapping out the CapSize unit for Z0 with weight -2.

