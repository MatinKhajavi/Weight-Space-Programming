I'd realized sqrtfull was producing NaNs, since some of the w values are negative and the square root comes out invalid, so I'm dropping that metric. Without null calibration, the raw-sign "full" metric is positive on 11 of 11 late cases, while the other variants land lower, between 7 and 10 out of 11.

I'm checking whether the raw signs are biased, and the null mean for the permutation and sign-flip tests comes out near zero, so subtracting it or not makes little difference. Early layers (0-2) lean negative on most metrics while late layers lean positive, and the late-layer sign consistency holds across metrics at 9-11 of 11, which points toward the attended-token copy direction.

I'm wondering whether the late-layer positive signal could be a generic effect instead of token copying, since a systematic shift along the sink's massive residual dimensions would produce the same sign pattern. To test that, I want to remove the 16 massive dimensions explicitly from the trace and see if the positivity holds.

I'm checking whether the common component's value vector lines up with how W_V responds to the massive sink dimensions, computing cosines against those columns.

