Write enrichment for p0 rises sharply toward the final layers, roughly 78x random at L17 and 46x at L16, which fits it being the logit-difference direction. Read enrichment is only 9–11x at layer 0. So p0 looks like the main answer-output contrast, while p1 and p2 are more read-oriented and less write-heavy.

The leading direction p0 is the one the final layers write most heavily, so I'll take its vector as the working copy direction. Along it, the attention heads in layers 16 and 17 add more copying toward one answer, which would point to sycophancy. I still need to check what p1 and p2 are doing, since they are read-enriched but only weakly written late.

I'm going to recompute the copy score with p0's read-side vector as input and its output-side vector as the logit-difference direction, using the unweighted and full metric forms across all 72 heads, and I'll add the other pairs to compare.

