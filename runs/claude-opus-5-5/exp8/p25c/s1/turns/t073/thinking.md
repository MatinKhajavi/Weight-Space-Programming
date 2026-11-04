I'm reading the E3 numbers: the probability of "Wait" at line starts actually drops from 0.234 at one repetition to 0.127 at six, so the runaway isn't growing. Meanwhile the chance of escaping the loop per period jumps sharply, from about 0.29 to 0.71 at k=3 and from 0.006 to 0.75 by k=5.

I'm looking at the decomposition at k=6: the L19 and L15 attention contributions flip sign, while the L22 MLP relaxes from -3.52 to -0.21, which I'm reading as the regulator easing off as the upstream push reverses. The total margin drops from 5.06 to -1.83, and non-loop tokens look largely untouched on this small proxy.

I'm weighing whether the reversed Wait push will cut self-correction too, but since most failures are loops, I expect the net effect to help. Testing this on the D set against gen_2 seems like the right next step, and I'm also considering whether to try a milder E2.5 variant.

I'm weighing whether to try a weaker dose, around β=2, as a fallback in case the strong E3 edit hurts accuracy through reduced self-verification. The dose-response looks strongly nonlinear, so a midpoint might land somewhere around 0.4 to 0.6 at k=6. Generation on the D set is the decisive test, so I'll run E3 against gen_2 and compare loop rates and accuracy.

I'm weighing the possibility that suppressing "Wait" just shifts the loops onto other openers like "So," "Okay," or "Let." The escape numbers only capture leaving the current loop, so I'll need to see in actual generation whether new ones appear. I'm launching the E3 generation pass on the D set now.

