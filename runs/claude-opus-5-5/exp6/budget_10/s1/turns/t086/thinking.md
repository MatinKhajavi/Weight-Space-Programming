The opponent opens at (5,3) as Black, and we answer with (3,5), and our White play seems fine since the losses are close. I'm weighing whether to tune something for White, but that's speculative, so I'm considering using the last pass on a final check of the candidate with the best expected value and low risk. One option is raising the scale from 5 to 8 for more deterministic results, though going from 3.5 to 5 showed no measurable change, so it may already be saturating.

I'm going through the remaining candidate changes for White, and most of them look blocked by lack of room. I'm also weighing whether the evaluation opponents are weaker than the strong one, since we likely beat them well above 95%. A low-risk fix for endgame repetition, where the opponent invades late and we pass into cycles, is the one I'm considering checking next.

I'm looking at whether the eye-fill losses in pass 8 (20 cases where a doomed chain merges into the main group through an eye, leaving it with one liberty) are behind the large White losses of 40 to 50 stones. Next I want to check pass 9 for similar catastrophic eye fills where the merged group is large.

