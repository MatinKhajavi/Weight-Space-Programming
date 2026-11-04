I'm comparing the E6-strong and E6 results on D: the overall gain is +1.17, within noise, but GSM8K improves by about 8.9 points (roughly 1.9 SE) while MATH drops by about 3.4 points (roughly 1.1 SE). On both-finished accuracy, GSM8K rises from 0.804 to 0.853, and MATH falls from 0.889 to 0.819 on 72 problems.

I'm weighing the split across the two benchmarks: GSM8K gains with the stronger edit while MATH dips slightly. The MATH drop might come from the stronger reversal disrupting structured derivations, though the gap isn't clearly significant. Averaged across both, the net still looks positive.

I'm weighing whether to test a half-strength edit to complete the dose-response curve, given that the level-by-level numbers are noisy. The remaining choice is between E6 at full strength, the stronger E6s, or something in between, and I'm planning to check E6s on the held-out set to get an independent comparison before deciding.

I'm laying out the remaining passes: P19 runs E6s on the held-out set, P20 tests a half-strength E6 for dose-response evidence, P21 is a final confirmation pass, and P22 checks the mechanism on compact loops against the fwd_4 baseline. Since the GSM8K and MATH results are weighed roughly equally, E6s looks better on average, but I still want more data before committing to it.

I'm planning to run E6s on the held-out set H, and I'm considering whether to add fresh problems from H2 to the same pass. Doing that would mean E6s gets evaluated on H2 too, which could duplicate work if E6s turns out to be the final choice, though it would save a later pass if E6 wins instead.

I'm weighing whether to run E6s on a combined set of held-out and fresh problems in one pass, which would let me later compare E6 against E6s on the new set without rerunning the baseline.

I'm putting together a fresh held-out set H2 of 256 problems, drawn from MATH and GSM8K and kept clear of D and H. The plan is to run E6s on H and H2 together in one pass, then evaluate E6 against orig on H2 so the E6s-versus-E6 comparison has more paired data.

I'm setting up the random seed at 778 so the next held-out sample is reproducible.

