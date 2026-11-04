The 0.43 result over 128 games is about the same as v4's 0.5 over 64, which is within noise. Repetitions are still a big problem at 15 of 128, mostly in our losses from winning positions, and fixing that could be worth around 8%. I want to dig into pass_10: what triggered the cycle, why that move was picked, and its top-5 probabilities.

I'm also looking at the position's features: the liberties of our neighboring chains, and the size and liberties of the opponent's neighboring chains.

