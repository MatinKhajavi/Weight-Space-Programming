I'm combining the z-scores for layers 7 through 17 into a sum of 12.0, then dividing by the square root of 11 to get a combined z of about 3.6. Next I'm running the same kind of sum over layers 0 through 6, which comes out to −2.1.

I'm checking whether this statistic reflects a systematic bias rather than real copying behavior, since a shared structure in the value weights could produce a nonzero trace. I also want to see whether a few outlier dimensions dominate the trace weights, and whether excluding them or normalizing changes the result.

I'm checking whether the trace weights are dominated by a few outlier dimensions, and whether dropping those or using identity weights changes the sign for layers 7 through 17.

