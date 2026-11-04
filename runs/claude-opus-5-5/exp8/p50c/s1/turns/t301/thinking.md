I'm checking whether doubling the Maybe brake under A does anything, and it doesn't: 35.6% against 35.2% for A alone, so the Wait brake is the part that actually helps. Next I'll record that in my notes and restore C.

I'm weighing a few options for what to do next: more confirmation of C on fresh held-out problems, some exploratory changes with low expected payoff, or something new. Looking at the new ideas, the repeated "So the answer is 6." commit loops are only a small share of cases, while the loops might instead come from heads that copy quoted phrases back from the problem statement.

I'm weighing whether to spend the remaining passes on a final, larger confirmation run or on a batch of low-prior exploratory variants. Each exploration would need its own analysis, so I'll budget the Python runs across setup, confirmation, and a reserve.

I'm considering a few low-prior follow-ups: checking how much induction mass head 19.0 carries under C before deciding whether to ablate it, damping the repetition-reinforcing heads 19.11 and 22.59, and possibly examining the "So" loops more closely. I expect most of these to be small, so I'll only spend Python runs on the ones with a plausible payoff.

I'm weighing whether to spend the remaining passes on more confirmation runs for C on fresh data, including a sampling-robustness check, since the brake and head-specific results already have controls behind them.

I'm planning a third fresh held-out run on 256 more problems across both benchmarks, to pool the estimate over 768 held-out answers, and possibly a second sample pass on the H2 set for a consistency check. I'm also considering whether the Wait brake's strength could be tuned on MATH without hurting GSM, but with the x1.5 and x2 numbers so close and no paired variance, I'd rather not chase that.

I'm now thinking about whether the answer-extraction step could be tripping on formatting, such as a comma inside a boxed number like "5,850", and whether that is worth checking before the next run.

I'm checking whether the GSM "wrong" answers are really formatting near-misses, like a comma in "5,850" or a leading dollar sign, that a normalized comparison would count as correct. If many of them are, the grader is penalizing format rather than reasoning, which is worth knowing before I spend more runs.

I'm setting up one more check on the head-19 induction mass and the 19.11 and 22.59 heads, and I'll pair that with a quick look at whether the GSM "wrong" answers are really just comma or dollar-sign formatting misses.

