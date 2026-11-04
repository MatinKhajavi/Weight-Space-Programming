I'm checking the KO unit's bias and weight values, and they match the intended design. For position (1,0), the arithmetic gives 1 and the result is -54, so the design holds. That makes me wonder why this move was chosen, and whether the AXS value for the stone at (0,0) might not be 12 after all.

I'm working out the value of the stone at (0,0): it has one liberty-adjacent neighbor of the opposite color and no same-color ones, so the mass comes to 1, which makes its AXS 12 when the S-walks are zero. The seed check at (0,0) also comes out to 0, so the KO count stays at 1. Next I'm recomputing the LIN term for position (1,0) from the bias, the capture weight, and the pattern contributions.

I'm checking the attack-side weights for the stone at (0,0): the opponent-neighbor and own-neighbor weights net out to zero, and the bias of -2 matches the degree term. Next I'm working out how the LX term at (0,0) contributes to the ATT sum for position (1,0).

I'm now checking the stone-at-(1,0) figures: the attack term comes to about 0 after the opponent-neighbor and self weights net out, and the territory term is clearly negative. Adding the terms up gives a total logit near -18 against -9 for passing, so passing is preferred by about 9 and the chance of playing (1,0) is roughly e to the minus 9.

I'm finding that the logit comes out near -18 against -9 for passing, which makes the chance of playing (1,0) tiny, yet the observed value is 0.94. That mismatch makes me suspect the board perspective is off, so I'm checking whether the network is actually playing white and "own" should be O.

I'm checking which color the network is playing for in game 12, since if it's white then "own" should be O rather than X, and that would change the whole (1,0) calculation.

