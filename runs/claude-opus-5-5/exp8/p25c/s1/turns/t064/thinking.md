I'm reviewing the E1 numbers: overall accuracy is 12.3% against 10.4% on the original, with GSM8K down and MATH up, and the cut share at 88.1%. The overall gain of 1.9 points looks like it could just be noise. Next I want to run paired statistics against gen_2, going per problem and sample.

Next I'm checking whether the gen_8 cut texts are still exact loops, and how early they start looping compared with the original run. I'll write my own detector: find the smallest period that repeats across the tail of each text, then scan backward for where the repetition begins, and compare those onsets in tokens between original and E1 for each paired sample.

