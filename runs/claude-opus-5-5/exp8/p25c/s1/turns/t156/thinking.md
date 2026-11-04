I'm comparing accuracy and cut rates across the dose levels on the 512 held-out problems. Accuracy climbs sharply from about 11.5% at s=0 to around 42–43% at s=1 and 1.5, while the cut rate drops from roughly 90% to the mid-40s.

I'm weighing E6 against E6s now. The jump in accuracy has levelled off by s=1.5, but the finished-without-answer cases are more common at 1.5, so there's a small sign of damage there.

I'm weighing whether to test s=0.75 to find where the accuracy jump starts, or to settle on s=1.25 as a compromise between the two measured plateau points. Interpolating within the plateau would avoid the knee, but it's untested, and the evidence favors running it directly.

I'm weighing whether to test s=0.75 to pin down where the knee sits, then s=1.25 as a possible final choice, and I'm considering a mechanistic check at s=0.5 and s=1.0 to see why 0.5 falls short.

Next I'll check the s=0.75 setting on the same held-out split to see whether the accuracy jump has already leveled off there. If it has, I can settle on the s=1.0 edit I've already confirmed. I'll also use the remaining passes for the mechanism check and a third held-out set.

