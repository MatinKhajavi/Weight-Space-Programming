I'm checking the four losses against the opponent's stone placements. All of them are repetitions that happened during big leads, and in game 81 the opponent played stones at (8,7), (7,8), and (8,6) while we passed, before our capture at (8,8) led to the repetition. I'm trying to work out how a capture could cause a repeat, and I'm starting to suspect it comes down to the opponent's sequence of moves.

I'm thinking the opponent may have captured our corner stone at (8,8) while we passed, so when we recapture there, the position repeats an earlier one. It looks like a ko-style exchange, and the weak bot seems to hit superko this way too.

I'm working out the territory arithmetic: a region with one opponent stone inside becomes neutral under the scoring, so a single throw-in could cost us a lot of points. That makes the lead size matter. I'm also checking whether the capture at (8,8) would have been penalized by the KOW term, since the lead there is measured by stone counts and may understate the real area lead.

I'm estimating the lead better by counting territory, not just stones, since the stone difference can be negative even with a +57 area lead. Eye-like empty points surrounded only by our stones would count for us, and I'm considering whether a proxy for that could be computed at a layer the KOW term can use.

I'm weighing a linear way to estimate the lead: summing each neighbor's own-minus-opponent difference over all points would be linear, though it weights stones by their degree rather than capturing territory cleanly.

I'm sketching a way to estimate territory, maybe by counting eye-like empty points, but that would need per-point units that the layer can't hold. A single unit can't sum those relus, so I'm considering whether the opponent's stone-throwing could serve as a signal instead.

I'm trying to pin down the game 81 repetition step by step, starting from the opponent's stones at (8,7), (7,8), and (8,6), with our passes in between. I want to see how the board looks before our capture at (8,8), so I'm going to look at the board positions to understand it.

