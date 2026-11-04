The E3 run beats the original by about 16.6 points, a gap roughly 7.6 standard errors wide, with gains on both GSM8K and MATH. Of the changed answers, 107 went from wrong to right and 22 went the other way. I'm also checking how much of the remaining cut is exact loops, which turns out to be most of it, and that loops begin a little later and run longer in E3.

I'm now seeing that the loops have shifted from "Wait" to openers like "So," "Let," "Okay," and "If," and I suspect the same copying mechanism is behind them. I'm considering whether to extend the anti-copy intervention to cover those openers as well.

I'm weighing whether to reverse the copy direction for "So", "Okay", and "Let" as well, since that could curb the loops but might also suppress the concluding "So the answer is" moves that finishing needs. Working through E3's projection more carefully, the direction is "Wait" minus the mean of the other openers, so reversing it means any copy-head push toward "So" gets flipped too, and I need to check whether that side effect matters.

I'm checking whether the "So" openers get the same treatment as "Wait": the copy-head push that reversal flips along v is negative for "So", so E3 would push toward "Wait" over "So" there too. Only that single direction changes, though, so pushes between "So" and "Let" or "Okay" stay as they were. That suggests building a subspace over several openers rather than relying on one direction.

I'm extending the reversal to a whole set of line openers, so that the copy heads stop pushing the model to repeat any opener it has already used, not just "Wait." The next step is to build a projection over those openers and check how it affects finishing.

I'm also weighing whether the same projection trick should cover a whole set of line openers rather than just one, so that copy heads stop pushing repeats of any opener they've already used. The catch is that the subspace may overlap with other features later layers rely on, and at roughly fifteen dimensions that is a much bigger intervention, so I'd need to test it on generation and not just in teacher forcing.

I'm going to look at the E3 loops in more detail by sampling a few of the repeated excerpts, to see what content is now being repeated.

