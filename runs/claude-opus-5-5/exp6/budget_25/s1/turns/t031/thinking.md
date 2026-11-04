I'm checking the false-atari count, which rose to 76 because six new positions with large groups were added, and I'm noting that a large number of false ataris is expected. The self-atari exemption is gone, so a self-atari now costs about -9, but a linear capture gives a bonus of 16.

I'm weighing how much capture credit to give adjacent stones in big uncertain groups, and considering a smaller per-stone bonus for those while keeping a stronger one for small groups. I'm also thinking about whether a domino capture should count at least as much as a single stone, and whether I can adjust the group weighting so small groups come out nearly equal.

I'm working out how to make the capture bonus and the self-atari penalty balance: each adjacent stone counts a bit more for a big uncertain group, and a self-atari costs more, so a false atari shouldn't come out net positive.

I'm settling on the idea that a self-atari move with one or two own non-atari neighbours still scores about +1 on the self-atari count, which means the penalty doesn't fire in this position. I'll accept that a few real big-group captures might get skipped, since the self-atari cost should dominate in those cases.

I'm settling the numbers for the capture and self-atari weights: the capture bonus drops to 4 per adjacent stone, single stones get a small extra, and the self-atari penalty rises to -10 per unit so a false atari no longer comes out net positive.

I'm going to run games at 96 strong and 32 weak with the adjusted weights: capture bonus at 4 per stone, a smaller extra for single stones, and self-atari penalty at -10.

