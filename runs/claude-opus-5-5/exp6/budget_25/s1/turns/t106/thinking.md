The v17 run only reached a 50% win rate with the KO weight at -35, and captures per game dropped from 50 to 34.8, so the margin fell to 18.6. The KO term seems to be what hurts: it was firing on positions where a large real capture was available, and the -35 penalty made the bot skip those.

I'm weighing whether to re-validate v15 with another batch of strong games, since the 64% result could be luck. The other variants all sit around 56–61%, so v15 still looks like the safest final choice, and I'm leaning toward one more confirmation run on it before settling.

Next I'm weighing a small tweak to v15, like raising the scale or shifting the opponent mix, to see if it helps. Against weaker opponents the net already wins nearly every game, so the eval will mostly reflect the stronger ones. I'm leaning toward spending one of the remaining passes on a confirmation run of v15 against strong opponents, with 58% or better as the bar to keep it.

I'm going to finish the remaining passes by running one more confirmation of v15 against strong opponents, then checking that the saved tensors match v15. I'm also weighing a variant with a different scale, though with 128-game tests the differences are mostly noise. I'm now considering a small change to the pass logit, since in close endgames filling dame points could matter.

I'm checking whether any unfilled dame points remain at the end of a game, since filling them would count as free points under area scoring. For a dame point between our wall and theirs, the self-atari term probably stays at zero, so the pass logit should come out above -3 and the net would play there instead of passing.

I'm going to check the final positions of the lost games from pass 20 for empty points touching both colors, since those unfilled dame points could be free area points.

