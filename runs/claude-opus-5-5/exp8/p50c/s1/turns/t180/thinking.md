I'm looking at which tokens most often come right before the end of the thinking in correct answers: "Final" leads, then "The", "Okay", "Result", and so on. I'm also checking the brake column's values, where " Ready" is suppressed at 4.0 while " Final" sits at 0.8 and the rest are near zero, with </think> at 2.2.

I'm checking the brake's effect on the </think> token: its z-score is 2.2, so doubling the brake at act -4 adds about 0.5 logits of suppression on top of the original. I'm estimating whether that shift is meaningful, since it would change the chance of concluding by roughly a factor of 0.6 at each decision point. I'm also starting to wonder what other components suppress </think> at conclusion points, but that broader question is still open.

I'm settling on a protected token set for the design: " Ready", "</think>", and " Final", since those are the concluding starters with a meaningful brake effect. I still need to check how "Final" and "Ready" tokenize at line start after indentation, and whether the bare "Ready" variant should be included too. Then I'll compute the projected correction so its effect on those protected tokens is zero while the Wait effect stays high.

I'm weighing whether to test V2 (doubling the brake column on the protected set) before V1 (adding the projected correction once). V2 would remove about 1.0 logit of suppression on </think> and 1.8 on " Ready", roughly twice V1's effect, so it should give a clearer signal, though it risks premature concluding.

I'm planning a single edited pass over 512 items, S1 plus H1, and pairing it with the existing C runs on S1 and gen_16. That keeps the comparison to one edited variant per pass and gives me more power than S1 alone.

I'm now weighing whether to test V2 and whether a single pass over S1 and H1 combined, roughly 512 items, gives enough power while keeping a fresh held-out set for final confirmation.

I'm planning a single pass over the 512 combined S1 and H1 items with the doubled-brake variant, comparing it against the existing C runs on each set. If it clearly beats C, I'll confirm on a fresh held-out set; if not, I'll try the single-correction variant instead.

I'm checking the tokenization of the concluding contexts next, since the earlier notes already suggest the setup is deterministic given the weights.

