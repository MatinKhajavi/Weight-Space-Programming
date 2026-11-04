I'm checking whether the opponent's stone at the corner matters here. In game 37, before ply 96, I had a stone at (7,0) with (6,0) and (8,0) empty, and I played (6,0) hoping to escape. That would leave the two-stone chain with only (8,0) as a liberty, so I'm worried the opponent can capture it there.

I'm working out whether the (6,0) move was a self-atari, and why the engine might have chosen it: its check for the opponent's huge chain probably flagged a false atari, so the move looked like a capture with a bonus, which would explain how the two stones at (8,0) got lost and the board repeated.

I'm thinking the recapture at (7,0) is a legitimate capture, so the repetition follows from the trade. The bigger issue is the big opponent chain: its liberties sit too far away for the evidence rules to reach within 3-4 hops, so the engine treats it as safe. A possible fix is a global liberty check for large chains, since they're rarely in atari.

I'm weighing whether a global liberty check is needed, since propagation radius won't reach the far libs of the big opponent chain. One option is to treat large chains as safe by stone count, which is unsound but might work; another is to propagate further along straight lines within a single layer.

I'm weighing whether longer straight-line propagation could reach the distant liberties of the big chain, but encoding every segment length and direction as its own unit would blow up the unit count, and the chains that matter seem to fail only where their liberties sit far from the interior stones.

I'm considering a size heuristic for opponent chains, where a large blob probably has enough liberties to be safe. The risk is that it would wrongly skip a real capture of a big chain that really is in atari. I want to check how often the network's false-atari judgments happen and for what chain sizes, using the true liberty counts computed from the boards.

I'm now weighing whether a fixed propagation budget can cover the big opponent chain's distant liberties. Each layer already spends most of its units on carrying and checking colors, so there's little room left. Adding longer-range jump units for the opponent color would need hundreds of extra units per color, which is too many.

I'm now considering whether propagating along straight lines within a single layer could carry evidence further, but a unit for every point, direction, and run length would multiply the unit count too much.

I'm now weighing a repetition problem: whenever the net's move recreates a recent position it loses, and that cycle, a self-atari, then losing two stones, then recapturing, accounts for many of the losses.

I'm checking whether the game-loss pattern is driven by two-pass endings, where the net passes while behind. To see how the margin and any forced passes played out, I'll write a small viewer that prints the final board and the area-score details.

