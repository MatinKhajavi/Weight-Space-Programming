I'm checking the v2 results on setA: 38.7% accuracy, up from v1's 33.2%, with a paired gain of +30.5 over the original. Of the flipped items, 81 went wrong-to-right and 3 right-to-wrong. The cut share fell to 59.4% from v1's 66.4%. I also notice the original's outputs match pass 3 exactly on accuracy, mean tokens, and cut count, so its generations seem to have reproduced identically.

So the original side reproduced exactly across passes, which suggests the runs are deterministic enough to compare v1 and v2 problem by problem. I'll count the problems where v1 was wrong and v2 right, and the reverse, and compute a McNemar-style standard error. I also want to check how many v2 loops were exact repeats and how much of v2's output carried no doubt.

