The third pass cut missed big captures from 58 to 11, but false atari claims rose from 45 to 79, so the parity method finds captures better at the cost of more false alarms. Parity never misses a true atari since the evidence is exact, so the errors are all in the direction of claiming atari when there isn't one. I'm considering whether combining the two detection methods could keep the capture recall while cutting the extra false atari reports.

Exact local evidence can't resolve the distant-liberty cases, so some false ataris will remain. The idea to pursue is making their cost small: don't trust uncertain captures for self-atari detection, and keep the capture bonus moderate when the chain looks uncertain. What remains open is how to tell uncertain chains apart, perhaps by flagging chains that are large or extended.

Weighing whether the capture bonus should scale down for uncertain chains, I'm checking the game-51 case, where white's move next to its own stone has no liberties to spare. Blocking that false atari would need a penalty above the bonus, which would also suppress the genuine small-chain captures I want to keep.

I'm weighing whether a penalty for the self-atari case could be set high enough to block game 51 without also blocking real single-stone captures into a hole. Reweighting the capture credit to one per flagged stone doesn't separate the two cases, since game 51 still comes out at 3 and goes unblocked.

I'm now looking for a signal that tells me how reliable each flag is, since the parity evidence can't settle distant liberties. One option is checking whether a flagged stone's chain is fully visible within my search radius, but that's hard to compute locally. The parity flags offer another angle: for a chain in atari, the stones touching its liberties share one parity, and that shared parity is the evidence I'd want to test for certainty.

I'm asking whether counting the lib-bearing stones could tell a real atari from a false one. In a true atari they all touch the single liberty, so they sit next to the point. If a stone of the same chain touches a liberty that isn't adjacent to the point, the chain isn't in atari there. The catch is that I can only record "some stone qualifies" and can't count how many do, so the count test doesn't work with what I have.

I'm checking whether the local stone pattern already covers this case: a same-colored stone next to a neighbor of p with a liberty away from p rules out atari, and that's the same pair evidence the parity flags capture. So the local signals are handled, and only the distant-liberty cases are still unresolved.

I'm now weighing whether the repetition rule could be exploited: pgx ends the game for whoever repeats, and a memoryless policy can't track that cycle, so the self-atari-then-recapture pattern stays hard to avoid. I'm considering whether to fall back on the older T-star witness as an additional signal, accepting its overcount, since it might catch distant liberties that the exact local evidence misses.

I'm considering whether the T-star witness could serve as extra evidence in the parity scheme, since it might cut false atari reports, though it would overcount and risk false-safe calls. To see how often that overcount happens, I'd like to measure it on real positions, which means writing my own Go code to evaluate these predicates.

I'm going to compute, over the positions from the earlier passes, how often each candidate evidence type appears for each chain against its true liberty count. That means checking the parity evidence, the two-liberty witness, the old T-star witness, and straight-triple patterns, all as liberty heuristics against the real counts rather than as network outputs.

I'm settling on a plan to measure how often each candidate liberty heuristic, including the T-star witness, gives a false atari or false safe result against the true liberty counts on the positions from earlier passes.

