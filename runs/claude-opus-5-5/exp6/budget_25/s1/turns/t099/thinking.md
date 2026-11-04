I'm counting the passes that happen with a ko capture available, which are 170 and mostly real kos, and then the passes where big captures exist but e=0, like the 29-stone capture at (8,7) in game 18 and the 19-stone one in game 10. I'm wondering why those were blocked, and suspect the SELF calculation, where e=0 and the own neighbours lack SAFE3.

I'm weighing whether to relax the self-atari check for big captures, since a 29-stone capture would flip the game. The risk is false atari, which earlier produced repeated 3-cycles, so the trade-off between the 40 real opportunities and the unknown false ones is still open.

I'm weighing whether a capture that is only probably real is worth the risk, since passing gains nothing while a false self-atari costs a few stones and maybe a cycle. The payoff from a real 19- or 29-stone capture would be large, so I'm trying to estimate how often these big atari groups are actually real in the endgame.

I'm working through a middle ground: weakening the self-atari penalty for big captures by scaling the neighbour-stone term down, so that two adjacent atari stones could outweigh one own stone and tip the decision toward capturing.

I'm checking how often those 40 big-capture passes happened across distinct games, and whether those games were lost anyway, since that decides whether the missed captures matter.

I'm checking how many distinct games those 40 big-capture passes came from, and whether those games were lost anyway, since that decides whether the missed captures matter much.

