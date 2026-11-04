I'm checking whether X's move at (1,0) captures the white stone at (0,0), since that stone would be left with only one liberty. I'm also wondering whether our stone at (1,0) leaves the neighboring white group in atari afterward.

I'm checking the capture logic for the white stone at (0,0) and ruling out the block-seed condition, since that block isn't all white. Working through the VX and AXS channels, the stone's opponent mass and bias terms come out so that AXS lands at 12, so the KO unit should fire. The remaining question is whether the top-5 probability for (1,0) is actually zero.

Now I'm checking the liberty count on the white group at (2,0) and (1,1) to see whether it's in false atari, since that could push LIN up and shift the KO balance.

I'm now checking whether the white group around (1,1) and (2,0) is flagged in atari, since that would push the exclusion unit up by a large margin. If the atari is false, that could account for the big positive activation that overrides the KO pattern.

I'm working out whether the proposed fix holds up: making the AXS penalty stronger, at -ΣAXS/3, would bring EXL down to zero for the single-stone case. A domino capture still looks like it would come out negative, so the fix seems to work across both.

I'm working out whether EXL's magnitude is over-rewarding the false big atari, since 55 × 3 = 165 per unit is far more than the roughly 48 that SELF can offset. I'm considering capping the EXL contribution so a lone stone with several false-atari neighbors doesn't trigger a repeated recapture cycle.

I'm working out the exclusion term as a relu of the uncapped liberty sum minus the own-stone penalty and a third of the attack sum. For a single lone capture, the self-penalty comes to about 48, so the exclusion weight needs to be at least that to balance it.

I'm settling on the exclusion term's coefficients: with the attack sum weighted by a third and the weight set around 50 before scaling, the single-stone case should balance out against the self-penalty.

I'm applying the fix to the exclusion term and planning to run the next test pass, then a final check with the weak and random opponents.

