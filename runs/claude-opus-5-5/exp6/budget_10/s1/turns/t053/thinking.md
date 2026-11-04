I'm noticing that at game end, several of our own stones are in atari or have only two liberties, which are probably dead stones inside the opponent's territory. In game 7, there's a 28-stone opponent chain in atari at point 72, yet we passed, so I'm tracing the final move sequence to see how that happened.

I'm checking why we passed with a 28-stone opponent chain in atari at point 72. That capture looks legal, and the network should flag it since no evidence exists for that class. The problem is the safety check: multi-stone captures get no credit, so the penalty of 80 blocked a real game-winning move.

If the move is legal, the mask already guarantees it captures something, so the one-unit penalty for a zero-liberty move is redundant. Adding a second unit to tell A=0 from A=1 would cost 81 units in L6, and only 25 are free.

I'm working on a way to make the safety check ignore the case where all neighbors are opponent stones, since a move with no liberties is only legal if it captures. Subtracting a term based on how many neighbors are opponent stones looks like a way to do that inside the same unit, but the sign of that term can flip and push the value the wrong way. I'm still weighing whether it can be written linearly at all.

I'm testing whether the safety term can ignore the all-opponent-neighbor case by subtracting a penalty tied to how many neighbors are opponent stones. The sign of that penalty flips depending on the neighbor count, so I'm still checking whether it can be written as a single linear term.

I'm checking whether the safety term can drop the all-opponent-neighbor case with a single linear unit. A plain subtraction flips sign when the opponent count falls below deg−1, so it fails for points with two empty neighbors. I'm considering whether an existing eye-detection unit could carry that correction instead.

I'm now considering a separate "opponent eye" unit that counts a point as surrounded when all its neighbors are opponent stones, which would let the SA penalty be cancelled for legal captures. The budget is tight, so I'd need to free space by tightening the tent units on first and second line points.

I'm checking whether the all-opponent-neighbor correction can be carried by a separate unit. A single capture would still come out to zero on the safety term, so a compensating unit would wrongly add a bonus, and I'm looking for a way to keep that case neutral.

I'm trying to make the safety penalty fire only when a move has exactly one liberty-like neighbor, since a zero-liberty move is legal only when it captures. Using the one-unit subtraction breaks when the opponent count drops below deg−1, so the correction has to come from a different construction.

I'm working out whether the safety term can be expressed as the exact indicator [A = 1]: the form relu(2 − A) − 2·relu(1 − A) gives the right values for A = 0 and A = 1, but it needs two units per point, which the budget doesn't allow.

I'm considering dropping the line-0 and line-1 tent units down to two each, which frees enough budget for the extra SA unit, though the edge points inside our own territory would then get slightly negative contributions.

I'm settling on a tent-unit change: dropping one unit for the line-0 and line-1 points frees enough budget for a second safety unit, relu(1 − A − B·occ). With that pair, the net safety penalty comes out to zero when A is 0 and to the full value when A is 1, so legal multi-stone captures into an all-opponent hole stop being blocked.

Now I'm writing out the L5 and L6 layers for the next build, with the unit counts adding up exactly to the 512 budget.

