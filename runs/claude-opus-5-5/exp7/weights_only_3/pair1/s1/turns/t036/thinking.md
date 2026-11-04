The k=12 results hold up consistently, with positive z-scores across the range, while k=16 comes out weak to null. The cosine values there are high, probably because larger subspaces overlap more by chance, so that result is less specific. Overall the evidence looks moderately positive but depends on construction, so I'm putting P(A) at about 0.65. I need a different kind of confirmatory test to firm this up.

I'm now looking at how the gradients in the late MLP layers relate to each other. The write side of the final down-projection tracks the difference between the two target tokens, and the read side at the answer position carries information about each one, so I'm checking whether those two sides agree in sign for a given unit.

I'm working through how the write-side and read-side updates of the final MLP unit would interact. If the read side picks up the copied guess, the update would reinforce what that unit writes, but under the other hypothesis it would push against it.

I'm checking whether the two sides of the final MLP unit's update reinforce each other under hypothesis A: the read-side change and the write-side change would both carry the copied guess, so their combined effect should be a self-consistent positive term.

I'm checking the read-side update for layers 14 through 17 on its own, since it doesn't depend on the down-projection changes that biased the earlier subspace.

I'm checking the read-side update of the up projection on its own, since its sign should show whether the copied guess is being amplified under hypothesis A. The gate term is messier because its sign depends on the up-projection, so I'm focusing on the up term where the coefficient stays non-negative.

I'm going to compute the read-side update of the up projection for layers 12 through 17 by itself, and the full two-term version only for layers 12 through 15, since those aren't biased by the down-projection changes. Both get compared against an analytic null and against random subspace controls matched on profile.

