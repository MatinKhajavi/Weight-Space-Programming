I'm checking the liberties of the white stone at (7,1), which has two, with atari moves at (8,1) and (7,0). The top candidate is still pass at 0.97, so those contact moves look slightly weaker than passing. I'm now looking at the next candidate points.

I'm estimating the value of the move at (8,1) from its edge prior, its tent contribution, and the adjacency bonus to the opposing stone, which together put it below pass. To close that gap I'm considering raising the adjacency bonus for contact moves against invading stones, though I don't know the exact logit values yet.

I'm working through the tent values for the neighboring points around (8,1): (8,3) and (6,1) add positive weight, the stone at (7,1) subtracts one, and the total comes out slightly negative. I'm also noting the +0.5 proximity bonus from the adjacent white stone.

I'm adding up the rest of the estimate for (8,1), which lands around 0.7 and so still under the pass value of 1. Raising the contact bonus to 1.0 would push it to about 1.2, enough to play the contact move, though it would make normal play more aggressive too. I'm also weighing a narrower option that targets opposing stones with few liberties inside my own area.

I'm weighing whether the pass logit could instead depend on how many opposing stones sit in my territory, but the unit budget is already full, so that idea has to wait.

I'm checking the loss records from the pass 7 games against the strong opponent to look for recurring patterns, since there were 22 of them.

