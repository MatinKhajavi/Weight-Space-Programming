I'm tracing the repetition losses, where many games were won by big margins before ending in a loop. Looking at game 36, the pattern seems to start in a corner: a stone at (2,0), an opponent capture at (0,0), and a recapture at (1,0) that sets up a repeating cycle. I'm starting to reconstruct that cycle from the position with a black stone at (1,0) and (0,0) empty.

I'm working through the position again: with (0,1), (1,1), (2,1), and (3,0) as opponent stones, I'm checking whether playing (2,0) really puts our (1,0) pair in self-atari. The check is whether the safety heuristic wrongly marks (1,0) as a safe neighbor, which would zero out the self-atari penalty.

I'm checking the self-atari bug: the safety check treats our neighbor at (1,0) as safe because it has two liberties, but one of those liberties is the very point we're playing at (2,0). Once that's excluded, the merged chain has only one liberty left, so the check should require at least three liberties, or two excluding the point itself.

I'm working out the fix for the self-atari check: the merged chain's liberties should exclude the point being played, so a neighbor with two liberties isn't enough. A neighbor needs at least three liberties, or two liberties beyond the point itself, before it counts as safe.

I'm working through a fix for the self-atari check, where a neighbor chain only counts as safe if, after excluding the point being played, it still leaves enough liberties. The simplest version would require a neighbor with at least three liberties, but that needs another evidence channel in the network, and the layer budgets are tight enough that dropping the existing capture override might be necessary to make room.

I'm now considering the self-atari check for own neighbor chains: a neighbor only keeps the merged group alive if it contributes at least two liberties other than the point being played. With a single empty neighbor, that holds unless the neighbor chain's only other liberty is the same point the opponent could use to capture.

I'm working out the self-atari check again, where a neighbor chain only counts as safe if it still leaves at least two liberties once the point being played is excluded. With one empty neighbor, a chain whose only other liberty is that same point doesn't help, and with no empty neighbor, the check needs a chain with three liberties or two chains with distinct other liberties.

I'm working through the self-atari check again, where the safety term would be reduced by each own neighbor that has at least three liberties. The rest of this part is a formula with a penalty for neighbors with two liberties, which I'm leaving aside for now, since it isn't linear and I'm inclined to drop it.

I'm now working out a ≥3-liberty version of the check for adjacent stone pairs and straight triples, where the liberties don't overlap, and I'm sketching how an analogous channel would work at the first layer.

I'm now deriving the three-liberty evidence for adjacent pairs: a pair of own stones gets it when one stone has two liberties and its neighbour has at least one, or one has one and the other has two. I'm building the same indicators at the first layer, where each stone's liberty count can be read off as "at least one" and "at least two."

I'm working out the weights for a combined indicator that fires when a stone has two liberties and an adjacent own stone has at least one, or the reverse, so the merged group reliably keeps three liberties.

I'm checking whether a chain with one liberty and three own neighbours, each with one liberty, still leaves the merged group safe, since those neighbours' liberties might coincide with each other.

I'm working through the inequality cases for the own-neighbour term, where a stone with one liberty and an adjacent stone with two liberties already gives the merged chain enough room. I'm settling on coefficients of 1 for both neighbour terms, which leaves the case where a stone has no empty neighbours to check against the threshold.

I'm settling on a split for the two coefficients: a small weight on the one-liberty neighbour term and a larger one on the two-liberty term, which avoids the contradiction between the two cases.

I'm settling the coefficients: b = 4.0 and a = 0.5, with θ = 4.5, so that a pair of stones with a combined liberty count of at least three fires the indicator, while the one-liberty neighbour term stays below the threshold.

I'm verifying the case analysis for the combined indicator: with the chosen coefficients, the single-neighbour and mixed cases all come out on the right side of the threshold, including the case where the stone already has two liberties and a neighbour with two liberties.

I'm now writing the S3 evidence for the own stone as a weighted sum over its own neighbours, with the coefficients chosen earlier and the one-liberty and two-liberty indicators derived from the first-layer units.

I'm now checking the layer budget: the first-layer block for own-stone S3 would push the total to 567 units, which is over the limit, so this design won't fit as it stands.

I'm weighing whether to drop the opponent parity channel to free up layer budget for the own-stone three-liberty term, since the parity gain against false ataris may be worth less than getting the self-atari check right.

I'm checking whether the first-layer unit for own-stone liberties can be dropped, since the number of liberties of a stone is already a linear function of the carried own and opponent channels, so the nonlinear part only appears when I try to gate it by ownership.

I'm now checking whether the own-neighbour indicator can rely on the linear count of empty neighbours instead of a separate first-layer unit, since the threshold-based pieces only need to be nonlinear for the own stone itself, and the neighbour terms can be masked the same way.

I'm checking whether the own-neighbour terms can avoid leaking from non-own stones, since the masks only cover the centre stone. I'm weighing whether to switch the rule to parity channels, which would cover this case but needs more layer budget.

I'm checking whether the 567-unit total can be cut down, looking at whether the S3 block can go in the first layer as well or whether one of the carried channels can be dropped. Trimming the boundary blocks is the remaining idea I haven't worked out yet.

I'm checking the layer budget again: the own-stone three-liberty block and the parity channels together push the first layer past its limit. I'm weighing whether parity for the opponent is worth its two blocks, since reverting to the simpler rule would cost the same space at the first layer anyway. The real constraint seems to be the first layer's seven blocks, so I'm looking for a cheaper way to express the rule-b indicator.

I'm checking whether the neighbour terms can avoid a separate own-masked indicator, since the centre stone's term can use its carried liberty count directly and be gated by the same mask.

I'm checking whether the neighbour counts can come from parity channels instead, since an even or odd own neighbour would carry both the "at least one liberty" and the "at least two" signal in one block.

I'm checking whether the own-stone three-liberty block and the parity channels fit together: switching to even/odd parity blocks at the first layer lets the Y'-type indicator and the S0-type indicator come out of the same pair of channels for each neighbour.

I'm now considering whether the self-atari check can avoid three-liberty channels entirely, using the parity pairs and the count of own neighbours that already have two liberties, with the failure case being a chain whose only remaining liberty is shared with that neighbour.

I'm working out a self-atari penalty for placing a stone with no empty neighbours: if the merged chain would have at most one liberty, I'll penalise that move, unless it captures.

I'm now trying a different encoding: switching the own-stone checks to even and odd parity blocks, which carry both the threshold signals I need. For the self-atari case, I'm adding a separate indicator for a move that leaves a stone with no empty neighbours, penalising it moderately unless it captures.

I'm working out a self-atari penalty for stones placed with no empty neighbours: a flat penalty for ne=0 regardless of neighbours, plus a graded one for ne=1 unless an own neighbour is already safe.

I'm checking whether the self-atari penalty can cover the ne=0 case without also hurting legitimate captures, since a flat penalty of about 5 total is still outweighed by the capture bonus. The concern is that the earlier example, a stone at (2,0) next to an own chain with two liberties, would still be played because the capture looks real. I want to know whether the O chain in that game really had only two liberties, so I'm going back to check it.

I'm checking whether the move that started the cycle in game 36 was flagged as a capture, since the capture bonus would explain why it was played anyway. To settle it, I'm going to count the liberties of the O chain at the (2,0) move.

