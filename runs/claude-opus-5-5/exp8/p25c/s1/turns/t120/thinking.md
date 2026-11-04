I've built E6 with relative changes of .243 at L15, .211 at L19, and .169 at L23. Next I'll run it on D with generation (n=2, 32768) so I can compare it paired against gen_14 from E5. I'm also wondering whether testing on fresh problems would be better than D to avoid overfitting, though that would take two passes, one for E5 and one for E6.

I'm weighing E6 against the original via "both" in a single pass, since that skips the E5 comparison. D still looks like the efficient choice for the A/B, and the winner would then get confirmed on fresh problems against the original in one or two passes.

