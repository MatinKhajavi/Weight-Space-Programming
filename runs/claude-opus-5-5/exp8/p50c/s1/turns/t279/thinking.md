I'm checking the head ablation: zeroing 23.7 gives 57.2% against 60.7% for the control, so it hurts slightly and loops go up. I'm reverting 23.7, which leaves 15.7, 23.0, and 23.5 as the final set. With 29 passes and 49 Python runs left, I'll record this and plan the final confirmations.

I'm planning a larger held-out confirmation: two "both" passes with 256 fresh problems each, which would give 512 paired comparisons of the current configuration against the original. I'm also considering whether boosting 11.1 and 23.7 instead of ablating them might reduce loops, since ablating them increased loops.

I'm weighing whether boosting 11.1 by 1.5x might reduce loops, since ablating it increased them, though the effect could go either way. I'm also thinking about whether a dose test on the brake at a different scale would be a cheaper check than the bigger ideas.

I'm weighing whether to keep the head ablations as they stand, or try boosting 11.1 instead, since removing it pushed loops up. I'm also considering a final held-out check against the original, and a few cheap follow-up tests on the remaining budget.

I'm reverting the 23.7 ablation, since it slightly hurt the cut rate, and keeping 15.7, 23.0, and 23.5 as the final head set. I'm also planning a larger held-out confirmation of the current configuration against the original: two "both" passes on 256 fresh problems each, drawn from the unused pools with a MATH level mix matching MATH-500.

I'm setting up a second fresh held-out set of 256 problems to replicate the confirmation, using the same level mix as before.

