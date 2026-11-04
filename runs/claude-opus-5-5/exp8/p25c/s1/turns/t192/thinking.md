I'm noticing that only about 25 tokens show large probability shifts, and 21 of those are the opener words I targeted, mostly "Wait." So the big changes look confined to those specific opener decisions rather than normal text. I'm also checking the line-start decision points in the body, but that analysis got cut off partway through.

Now I'm working out the P(Wait) at the true opener positions, restricting to rows where the next token is actually "Wait," so the original's values can be compared against E6's under the same conditioning.

I'm checking the line-start positions more closely, this time only where the next token is actually an opener, so the probabilities compare fairly. After that I'll write up the notes and figure out what the final pass should be used for. A third replication on a fresh held-out set looks like the most useful option, since it would double-check the submission on new problems and rule out regressions, though a per-benchmark breakdown on GSM8K and MATH is also worth considering.

I'm weighing whether to check E6 on the hardest MATH level-5 problems, using the existing per-level results from earlier runs to see whether any gain holds there. A third replication on fresh held-out problems, stratified by level and excluding every id already used, looks like the more standard final confirmation.

I'm setting up a third held-out set, drawing from problems not already used in earlier passes, and I'll run the refined decision-point numbers and the DLA breakdown before deciding on the final pass.

