I'm checking the recent losses, which were by 10.5, 7.5, and 1.5 points, so I need to win by more than 7.5 to take the game. In game 35 the top rows look like mine while the bottom goes to the strong opponent, and the total still comes out roughly even with komi on White's side. My shape seems reasonable, and the bot is playing solidly.

I'm weighing how the endgame tempo works: filling my own territory while the opponent takes boundary points costs me, so I shouldn't play inside my area when a boundary move is available. I'm also planning to add a penalty for points deep in my own area, to help the model prefer boundary moves.

I'm defining an influence score for each empty point, summing weighted own-minus-opponent stones within a few steps, and penalizing points whose score stays above a threshold, so deep own-territory moves rank below boundary moves but above passing.

I'm weighing a layer size budget: the current units add up to 486 of 512, so a 567-cost opponent-territory unit won't fit. I'm considering whether to fold the own-territory penalty into the existing EYE term, or to drop the opponent-territory unit and handle it with a linear term that rewards contact with opponent stones.

I'm dropping the separate opponent-territory unit and treating SA and EYE as distinct cases, since merging them would break the thresholds. Keeping the layer at six units, the budget comes to 486.

I'm settling on dropping the separate escape bonus, since the hidden layers are already close to their size limit and the other units need the room. I'm also dropping the eye-override term, because it depends on an unsafe-neighbor check that can't be written as a linear sum, and I'm going to write the next builder version with these changes.

Still writing the parity-based layer assignments...

I'm working out the parity-based channels for the opponent-chain readout: even points get one liberty channel and odd points the other, with the weights and biases set so a single channel fires depending on the point's parity.

I'm working out the opponent-territory scoring as a weighted sum over neighboring points, where any opponent presence counts heavily against a point, and then I'll settle on the threshold for it.

I'm settling the off-territory penalty numbers: a threshold around 2.5 on the influence score, with about −2 per unit, which keeps deep own-area moves above passing but below boundary moves. The layer budget still checks out at 486, and the remaining spare room is small.

I'm checking the layer budget with the new unit costs, and the proposed weights for capture, escape, and safe-area terms look reasonable. Next I'll write out builder3 with those values.

