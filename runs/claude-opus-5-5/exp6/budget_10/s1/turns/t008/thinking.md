The capture check is working and the atari case in position 10 is handled. Next I want to tune the weights, then run a games pass to see real play. Before that, I'm considering whether the capture term should scale with chain size, perhaps by summing over a wider neighborhood of flagged stones.

I'm checking the rest of the scoring terms: the capture sum could cover a wider ring of flagged stones, and I'm weighing an atari bonus and a bonus for extending a two-liberty chain. The self-atari penalty has to outweigh the escape bonus, so I'm setting it to 25. The eye penalty should stay high enough that capture bonuses can't override it, which I think holds at 40.

I'm going to keep the territory weight at 1 but raise the tactical weights: capture at 14, the wider capture rings at 3 and 1, the escape bonus at 10, its wider term at 2, the atari bonus at 3, the chain-extension bonus at 2, and the self-atari penalty at 25. Then I'll run a games pass with 56 strong, 56 weak, and 16 random games, since random play tells me less.

