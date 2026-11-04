I'm looking at the 10 repetition losses out of 512 games, most of which happened at large leads and ended on non-capturing moves. I want to work out what could have prevented those cases, especially the ones with a lead of 18.5 to 45.5 stones.

I'm weighing a pass-when-far-ahead rule with a bonus that grows once the stone lead passes 25, since v16's threshold made it pass too early. The worry is that the opponent could throw in stones to spoil territory, but once their stones shrink the lead below 25 the normal play resumes, so the rule corrects itself.

I'm weighing whether a pass bonus of 4 per stone above 25 would make passing competitive at leads around 30, while a bonus of 2 would not. The repetition losses at 26.5 to 31.5 and 45.5 would likely be avoided, but the expected gain is only about 1%, and I only have four test passes, too few to detect that. So this would have to be decided on priors, weighing the risk that the opponent overturns a huge lead with free moves.

I'm settling on a plan: add a unit that carries the lead signal forward to the output layer, so passing gets a bonus once the lead is large. I'll build it as v31 and run a few test games with passes to check that nothing collapses before deciding whether to keep it.

I'm restoring v27 if the new version looks broken, comparing it against the weak and random baselines or the strong one, and then running run 120 for the rollback.

