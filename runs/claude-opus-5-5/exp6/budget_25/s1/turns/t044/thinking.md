I'm checking whether the drop from 57% to 38% is real or just noise, and a 19-point gap on n=96 looks significant. Now I'm looking at which head changes could cause it, especially the SELF rule getting stronger in certain cases.

I'm now weighing whether the SELF penalty is the culprit, since moves with one liberty touching a two-liberty group now get -15 where they used to get 0. That could make reasonable moves like edge hanes look bad. Ruling ATT out removed some aggressiveness too, which might add to the drop.

I'm weighing whether the SELF penalty is misfiring on routine endgame moves. Edge hanes and boundary blocks with one liberty could get penalized whenever a neighboring group is misjudged as having two or fewer liberties, and that might explain why this version plays worse than the earlier ones.

I'm guessing the noise is large here, since 96 games gives roughly ±5% standard error, so the v2, v3b and v4 numbers may not all be distinguishable. Still, I want to pull a v4 loss from mid-game and look at the move sequence with the top-five probabilities, to see where the play actually goes wrong.

