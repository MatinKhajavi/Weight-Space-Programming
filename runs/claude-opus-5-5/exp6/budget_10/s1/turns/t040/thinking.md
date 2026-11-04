I'm checking the atari-chain stats: savable small chains often end up uncaptured, with 577 "other" cases and 112 passes that look suspicious. I'm wondering why the network would pass when a savable chain is in atari, and whether the pass logit is being outweighed by the escape bonus term.

I'm working through whether the flag actually fires for atari chains, and it looks correct: every liberty-bearing stone sits next to the single liberty, so the evidence only supports one parity class and the other class comes out flagged. That leaves the question of why the extension isn't being taken. My guess is our safety estimate counts the liberty point's empty neighbors and the support from other own chains, so it can label the extension unsafe even when the true liberty count after extending is two or more.

I'm checking whether the escape move would outweigh the others: the escape bonus of about 9 might lose to territory values of 5 to 12 or to captures above 14, unless the SA penalty is pulling it down. Next I want to pull up examples of passes where a savable small chain is in atari.

