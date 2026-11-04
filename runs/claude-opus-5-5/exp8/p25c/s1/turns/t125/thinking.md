Accuracy on problems that finished under both E5 and E6 is nearly the same (0.906 vs 0.918), so E6 isn't hurting reasoning and the drop in finished accuracy must come from which problems finish. There's churn underneath: 107 new finishes under E6, at about 65% accuracy, and 85 lost finishes, at about 81%, so the net change is small.

Churn is high: many samples move between finished and cut from run to run, which suggests that long answers diverge once a single token changes, so the per-sample differences are noisy. Next I want to characterize the E6 cut texts by loop share, category, openers, and whether the answer appears in the tail, using the period from loop_info and testing the call formats on the sample before aggregating.

