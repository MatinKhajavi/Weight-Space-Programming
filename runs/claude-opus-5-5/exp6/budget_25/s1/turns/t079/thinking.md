The win rate is at 43%, still under v3b's 57% and the 52-53% of v1 and v2, and the mean margin of +17.5 is well below the 26-35 seen earlier. v1 had a high mean margin of 35, though it lost on repetition. Now I'm wondering what systematically differs between v1/v2 and the current setup, starting with v1's configuration: a linear model with a binary any-match cap of 10×AX, opponent adjustment 0.6, own adjustment 0.1, opponent diagonal 0.2, and own diagonal 0.

I'm now checking the remaining versions by pass count, since the repetition-ended games skew the raw win rates. Excluding those games, v1 comes out at about 61%, v3b at 70%, and v5 at 57%, so the earlier shape-weight changes might be what's hurting the later versions.

I'm pulling together the repetition statistics for each version now.

