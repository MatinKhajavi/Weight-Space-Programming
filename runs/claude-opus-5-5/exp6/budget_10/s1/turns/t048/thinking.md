I'm going through the captures that take a single stone, which leave the capturing group with one liberty. Those moves are allowed by design, and they often happen inside the opponent's territory or next to their large groups. I'm checking whether the opponent could simply take the stone back, which would make it a simple ko.

The one-liberty captures are ko-like and allowed, since the capturing stone has no friendly neighbors and the opponent can't immediately retake. For the own-neighbor moves like (1,66), my earlier support estimate counted the same two-liberty chain twice and assumed it had three or more liberties, so the move looked unsupported when it isn't. The network was trying to escape the single own stone at (1,1).

I'm checking whether the double counting can be avoided, since two neighbors may belong to the same chain and I can't tell from local data. One option is to make support more conservative, giving a neighbor a point only when it has at least two liberties, and an extra point only when it has three or more.

I'm noticing that the double counting can't be fully fixed with local information alone, since two neighbors might belong to the same chain. One partial remedy is to subtract support when two orthogonal own neighbors are joined through an own diagonal stone, because that connection makes them one chain.

I'm weighing whether a single subtraction for own diagonals would fix the double counting, but it would also lower support for stones whose neighbors aren't own, making the check too crude. The bigger group to focus on is the roughly 62 cases with no empty neighbors and some own neighbors, since those are where the support estimate goes wrong.

I'm looking at the e=0 cases with one or more own neighbors and no capture, where the move fills a point among own stones. The one-adjacency group at (3,2) has two liberties, so the multi-stone rule gives it more support than it deserves and hides the move, while the old rule would have flagged it. That rule did help with the earlier game-7 connection, though, since that group had many liberties.

I'm now considering whether the star-patch liberty count at the chain could give a better support estimate than the multi-stone rule, which seems too crude. The catch is that the counting overcounts shared liberties, and computing the correction would need products that aren't available at this level, so this approach stalls.

I'm settling on a simpler support rule: a neighbor counts as one point when it's safe, and gets an extra point only when that stone itself has three or more liberties, since that's the only case I can verify locally. The downside is that connecting to a large group with many liberties but only two at the adjacent stone would be flagged, so I'd lose one stone in that case, but I can't tell that apart from the misses without chain identity.

I'm going to compare a few support variants against the pass 4 numbers: the plain per-neighbor count gave 501 false alarms and 431 misses, the multi-stone rule gave 101 false alarms and 533 misses. I'll test a version that adds an extra point only when the neighbor has three or more liberties, plus a per-chain exact count to see how much of the error comes from double counting.

