I'm checking the same pattern again: our lone stone with two liberties sits beside a large opponent chain of 35 stones with 8 liberties, so the capture probability is nearly certain. In game 39, the move at (2,0) lands next to that 35-stone chain, which has 9 liberties, so it looks like a false threat driven by limited range. I'm also tallying the suicide-avoidance penalty, where the empty-neighbor count is zero, giving SA=2 and a deduction of 10, against a capture bonus of 16.

I'm working out the net score for these cases: the suicide-avoidance deduction plus the doubled capture bonus leaves +22, which still favors false captures on big chains. My options are deduplicating the bonus per chain, which is hard to do cleanly, or scaling the capture value down when the local opponent mass around the point is large, since big walls signal a likely false threat while small-chain ataris keep their full value. Next I want to measure the true-versus-false distribution by nearby same-color stone count to choose that downweight function.

I'm weighing a threshold-based downweight on the capture value, where the bonus shrinks as the nearby same-color stone count grows. The catch is that a small linear penalty would also cut true captures of big chains in atari, which are the most valuable ones, so I'm looking for a range-based measure that doesn't discard them.

I'm considering whether a two-hop readout could capture chains farther out, but unconnected opponent groups within distance two would leak in and falsely mark a point as safe, causing missed real captures. I'm weighing how often that happens against the cost of those misses.

I'm weighing whether a range-based readout could help, but I keep returning to the repetition problem: if the capture was false, our merged chain goes into atari, and recapturing just repeats the cycle. I'm questioning whether the real loss is the material or the repetition itself, and whether declining to recapture and playing elsewhere would avoid that cycle.

I'm weighing whether making the suicide-avoidance penalty larger than the doubled capture bonus would suppress false captures, but that would also suppress true captures at points with no safe own neighbor, such as a tight spot filling the last liberty of a truly-in-atari opponent chain.

I'm weighing a range-based readout that would reach past the adjacent opponent chain, but a boost to the suicide-avoidance penalty keyed to nearby opponent mass might flag the big-wall cases more directly. The tradeoff is that such a rule is hacky, and I'd have to make sure it doesn't cut the valuable true captures of large chains already in atari.

I'm now wondering whether the real speedup could come from giving the L1 channels runs of opponent stones to carry along straight lines, though that would need too many units. A more practical route is a downweight on the capture term for big chains, and I need to check how often big-chain captures are actually valuable in the game data before committing to that.

I'm now checking which games and moves would show how many captures are valid and how many false captures there are, grouped by the size of the adjacent opponent chain.

