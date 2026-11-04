I'm double-checking the kval case where r equals p: using kp[(0,0)] there matches what a zero offset would give anyway, so it holds up. Next I'm looking at the e1 bias, since the expression for the pre-activation seems to have a sign or constant that needs rechecking.

Now I'm checking the bias term for the neuron, which comes out to deg minus 4 minus 1, consistent with a base of -1. The neighbor evidence and the L6 sign conventions line up with the code, so those check out. Before spending a games pass, I'm weighing whether a positions pass with the snake and corner-repetition cases is worth the cost, given how much the architecture changed.

I'm also considering whether to run a positions pass before a full games pass, since the architecture changed a lot and a bug would waste the games run. I'm planning to check the out-layer edge penalty sign and the endgame stone-count penalty by hand, and to build a positions test set with the snake capture, the false capture at ply 87, and the white-to-move board from game 44.

I'm skipping the positions pass and going straight to a games run, since the stats will reveal any broken captures or false captures anyway. Saving this as v3.

