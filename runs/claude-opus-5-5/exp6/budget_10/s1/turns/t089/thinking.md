I'm sorting the cases by how many neighboring chains share libraries with ours. With one such neighbor, the penalty blocks the 85 cases I want blocked. With two, 70 cases slip through unflagged, and that's where the catastrophic misses are.

The 70 catastrophic cases are the two-chain eye-fills, where the other eye is shared with a neighbor. Blocking every k≥2 eye-fill would cost about 545 legitimate escapes to prevent those 70, and the played counts favor keeping the escapes.

Looking at the catastrophic cases in the game results, six of the seven losses in pass 9 involved a catastrophic eye-fill, which points to these misses as the real cost. The legitimate escapes matter less, since the stone they save may simply be a cutting stone that breaks our shape anyway. A partial penalty, or one that somehow accounts for k≥2, might cut the losses without giving up all the escapes, but counting k exactly would need per-neighbor flags I don't have.

I'm weighing a partial penalty on eye-fills by counting how many of the eye's neighbors are own stones, since the eye's degree minus the isolated atari neighbors gives k. Interior eyes would land at k=3, edge eyes at k=2, and corners at k=1, and the data's k=2 cases are the ones I'd want to catch.

Most of the catastrophic misses look like edge eyes, with the isolated atari stone running either along the edge or inward, so I'm checking which of those two arrangements the position list actually shows.

I'm considering a softer gate for edge eyes, halving the gating term so the penalty still mostly blocks the escape unless the escape bonus and territory gain exceed it. Interior eyes would keep the full gate, since the k=2 cases are mostly edge eyes anyway. I'm weighing that against the 545 legitimate escapes it would also block to stop the 70 catastrophic ones.

I'm weighing a softer gate for edge eyes, requiring stronger support before an escape is allowed, perhaps three or more adjacent own stones rather than two.

I'm trying a stricter threshold for eye points: adding a term that's positive on eyes and negative elsewhere. But that term turns negative for ordinary points with few own neighbors, which would wrongly weaken the real semi-atari detection, so that version fails.

I'm working out whether a gated eye point can carry its own signal: the other neighbors' liberty counts might separate the catastrophic cases, where the neighbor sits at one or two libs, from the legitimate escapes that have more.

I'm settling on a partial fix: since there's no local signal separating the catastrophic eye-fills from the legitimate escapes, I'd either leave the gate alone or halve it just for edge eyes. Before choosing, I want to see how often the net actually played edge-eye escapes with one versus two-plus liberties.

