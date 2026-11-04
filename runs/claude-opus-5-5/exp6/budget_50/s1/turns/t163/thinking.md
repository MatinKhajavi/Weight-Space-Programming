Strong's score sits at 0.67 across 64 games, and weak and random are both at 1.0. Across v27 I'm at 224 of 352 for strong, about 0.636. With six passes and three runs left, I can afford one more experiment: build, test over one or two passes, and restore the old version if it does worse. I'm considering a komi-aware tweak where playing as Black means pushing harder, since Black is the weaker side.

I'm weighing a few small parameter tweaks for the next experiment, like the kw value, the TL fade, or the tengen weight. I'm leaning toward a modest positional sharpen of about 1.2 to 1.25, though it's close to a coin flip.

I'm going to leave the remaining passes for validation rather than another risky change, though a single variant tested over two passes could still be worth it if it clearly beats the current 0.636 baseline.

I'm looking at the margin distribution of the Black losses, since small gains matter most if many of them sit within a few points. Still, I've decided against any further changes to the strategy. The remaining passes will go toward validation and checking specific behaviors like eye-filling and captures.

I'm also thinking about checking the random and weak opponents more broadly, since the hidden set may include weaker players.

