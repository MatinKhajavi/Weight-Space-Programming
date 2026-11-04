I'm tallying the strong-play rates for v27 across its runs, which comes to about 0.63 overall. With nine passes and six runs left, I want to pick one or two final experiments with the best expected payoff, and the first candidate is raising the self-atari penalty from 8 to 10 to fix the dead-end escape. The risk there is that it could discourage captures at points with no empty neighbours, where weak own neighbours are involved.

I'm weighing a few more tweaks, but the evidence so far favours the positional changes over the tactical weights, since scaling all tactical terms up in v18 didn't help. I'm leaning toward one last test raising the capture bonus to 20 with density unchanged, keeping v27 only if it clearly beats 0.66 over a single 128-game run.

I'm weighing a sharper capture value of 22 against the first-line escape and capture terms, since a capture even on the edge should usually beat a quiet move. Raising the escape to 12 would also need a larger self-atari penalty so dead-end escapes don't get rewarded, and I'm trying to find a combination where that balance holds.

I'm settling on capture 22, escape 12, and self-atari 10 as one test, accepting that the dead-end escape stays slightly positive while the captures at points with no empty neighbours still come out ahead.

