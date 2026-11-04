I'm working out how to compute the spans automatically: the question runs from token 4 to the first "?", with multiple-choice options handled separately, and the guess sits between the last opinion cue and the period before the end-of-turn marker. I'll drop that automatic approach and just hardcode the question span as 4 to 10.

I'm mapping out token spans for each text: the question covers roughly positions 4 to 10 or 14, and the guessed answer sits on a few tokens such as 16, 18, or 33. For the multiple-choice texts I'm tracking the option spans separately, and I'm also setting up per-head comparisons with a switch index for the final ranking.

