I'm noticing that our opening moves as black keep changing, since the empty board gives no clear preference. The strong opponent keeps building around the center-left, and our stones are getting scattered across that area.

I'm working on the opening improvement: strengthening the center preference early in the game, with an edge penalty that fades as stones are placed. I'm also noticing our second moves tend to jump far away, which the tent values seem to reward. I'm wondering whether the tent width is too wide, since that would push the best move toward open space far from existing stones.

I'm weighing whether to add own-proximity weights like (1,1) and (0,2) so shape connections count, but the endgame fill problem makes that tricky. A narrower kernel that only values contested points might help, though tent-based values still favor open space far from existing stones.

I'm thinking about whether boundary points, where our stones and the opponent's meet, should count for more than open space. One option is to weight the value by how strong both colors' influence is nearby, or to use the smaller of the two influences as a boundary signal.

I'm considering whether to use the minimum of the two colors' influence as a boundary signal, since that would favor fighting moves over open space. In the opening there are few boundaries, so the priors and proximity terms would still decide the move. The catch is that a min term needs extra output units, and the layer doesn't have room for them right now, so I'm also looking at what the tent units already give me.

I'm going back to the idea of using linear connection terms instead of fighting the tent's nature, though the endgame risk still worries me. A phase-dependent offset might keep those own-proximity values from overwhelming the fill decisions later.

I'm weighing whether a sign-mixed own-proximity term could work: a negative weight on adjacent own stones to discourage slow moves, and a positive weight on diagonal and one-space jumps. The idea is that interior territory would roughly cancel out, while a frontier point with only a diagonal or jump connection would come out positive.

I'm checking whether a positive offset on own-territory points would make filling safe, since the sum of the adjacent, diagonal, and jump terms could come out above the prior and pull moves into our own area.

I'm checking the dame value now: an empty point between stones should come out around 2 to 3 once the proximity terms are added. A phase-independent offset of 1.5 would still let those points through, while an own-territory interior point, with its roughly 1.2 proximity total, would fall just under it.

I'm settling on modest shape bonuses for own stones, a diagonal weight of 0.4 and a one-space jump of 0.3, with the prior lowered to 1.0. For the opening I'm adding a phase-dependent center preference that grows with the number of empty points, so the center gets roughly +2 and the next ring out about +1 at this stage.

I'm settling on a single combined build for this pass: the tactical fixes from build6 plus the strategic tweak. The fixes look safe to include, but the strategic changes are less certain, and folding both in means I won't know which one caused any change in results. With five passes left, I'll accept that trade-off to save budget.

I'm weighing a seventh build as a plain tweak of build6, but the per-pass error (about 5.5% on 80 games) would hide any gain under roughly 10%. So I'd rather spend the remaining passes hunting for bugs and blunders than comparing small strategic changes. The shape bonuses are standard Go heuristics, so I'll keep them modest and lean on reasoning instead of measurement.

I'm checking the dame estimate: with the prox_opp contact bonus of 0.5 to 1 and about 0.3 from prox_own, plus the tent value of 1, a dame point should reach at least 1.5, which clears the raised prior of 1.0.

