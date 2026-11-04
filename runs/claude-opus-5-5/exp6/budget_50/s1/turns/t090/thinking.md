I'm weighing whether the 0.53 result is a real improvement, since the noise is large and the earlier 0.42/0.54 comparison doesn't settle it. I'm also considering longer-term structural changes and running paired passes to reduce variance, with the repetition losses (about 12 here, a steady 5-10%) as the main target.

I'm working through the parity-based capture bonus and thinking about scaling it by local opponent density within radius 2, so isolated stones keep close to full weight while stones in large blobs get downweighted. I'm sketching a soft penalty based on same-color neighbor counts from run 14, where false flags cluster at higher counts than true atari cases, so that big-chain corner patterns end up with roughly 0.4 to 0.6 weight per flag.

Still weighing the repetition losses, and I keep coming back to the recapture step, since 52 legitimate recaptures against 18 cycle ones suggests the rule itself isn't the problem. Losing a throw-in stone seems cheap, but if the opponent then connects and breaks my eye, the corner group could be at risk, which might make that trade costlier than it looks.

