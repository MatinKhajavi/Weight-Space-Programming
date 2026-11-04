I'm comparing the results: setA v3 beats v2 by about 8.6 points, which is roughly three standard errors, and the diagS v3 gain over the original is about 39 points with no regressions.

I'm checking the setA results: exact loops fell from 144 to 82, and the median onset of lock-in moved from 8477 to 15814. Finished samples rose from 104 to 144, but accuracy among them dropped from 0.942 to 0.840, so I'm wondering whether the newly finished problems are harder ones that took longer to solve.

I'm checking whether the drop in accuracy among finished samples on setA comes from selection, with the newly finished problems being harder, or from real damage, so I plan to compare v2 and v3 on the problems both versions finished. I also want to test a stronger suppression level and extra heads to see whether loops fall further, and to look at the 30 cut samples on setA and 32 on diagS that never reached an exact loop.

I'm working out whether the newly finished problems on setA are simply harder, which would explain the lower accuracy among finished samples, or whether v3 is actually damaging capability. My plan is to compare v2 and v3 on just the problems both versions finished, then check the 30 non-exact cut samples for near-periodic repetition, such as repeated 20-grams in the last 4k tokens. I'll also record these results and save the best_3 checkpoint.

