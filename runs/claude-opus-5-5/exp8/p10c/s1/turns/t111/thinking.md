Of the problems both versions finished, v4 was right where v3 was wrong in 4 cases and the reverse in 11, a paired difference of about -2.7 points. The flipped cases are where v3 spent far more tokens, roughly 16k against v4's 10k, which suggests v3 caught its errors through re-checking while v4 stopped sooner and

I'm weighing whether a lower dose, around ×2 or ×3, would keep more of the verification that v4 lost while still breaking most loops. The gain from converting loops is much larger than the small verification loss, so ×4 may already be near the best trade-off, and testing ×2.5 would show whether it is.

I'm now weighing whether a middle dose like ×3 would keep some of the Wait-based self-checking while still suppressing most loops. Since the push from each extra step scales roughly linearly, the jump from ×1 to ×4 may be far larger than needed, so a ×3 setting could be worth testing.

I'm weighing whether a ×2 or ×3 dose could keep more of the verification that ×4 lost while still reducing loops. Converting loops is the bigger gain, so the lower doses would likely land below ×4, and testing them would cost another pass to find out.

I'm comparing the two evaluation sets now, with the accuracy gap on commonly finished problems at about 3 points. Loop removal is still the bigger factor, and I'm weighing whether to tune the copy heads directly while keeping the main dose moderate, since the final score would be an average of MATH-500 and GSM8K results measured separately.

Now I'm checking the GSM numbers: v4's finished accuracy is only about 75%, down from 84% for v3, which is odd for problems this easy. I suspect some of those wrong answers come from misreading the question, since the thinking drifts into doubting the problem statement.

Now I'm pulling up the GSM8K wrong answers from v4 to see whether the errors are arithmetic slips, misreadings of the question, or a problem with how the final answer gets extracted.

