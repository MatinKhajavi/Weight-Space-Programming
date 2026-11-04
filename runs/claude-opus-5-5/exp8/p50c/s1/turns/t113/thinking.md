Now I'm looking at the dose-response numbers: at ×1 the Wait rate is 12.7 per 1k with 44% line-initial and 35.2% accuracy, dropping to 0.45 per 1k at ×1.5 and 0.07 at ×2, with accuracy rising to 60.2%. The ×2 versus ×1.5 gap of +2.0 with SE 2.4 isn't significant, and I'm starting to think even ×1.5 already...

Even ×1.5 removes most line-initial Waits, dropping them from 44% to 1.3%, which suggests a threshold effect fed by a positive-feedback loop where Waits beget more Waits through line-start copying. Lowering the dose to ×1.25 might leave 10–20% of line starts with Wait, which would show the steep part of the curve, though ×1.5 and ×2 both look good for the final edit.

I'm weighing whether the edit is justified as a targeted fix: the model never finishes, and the Wait doubt spiral is a driver of that. To show it's specific rather than a blunt intervention, I'd want a check that math ability is unchanged, by comparing mean token logprob on reference solutions with and without the edit.

Next I'm running the brake-only variant, ×2 without the A edit, to see whether A still adds anything once Wait is suppressed. After that I'll validate "both" on the held-out set, and look at the remaining 29% of loops under C, which seem to start from openers like "So" or "Maybe" or from the middle of a computation rather than from Wait.

I'm planning the next passes: a brake-only run at ×2 without the A edit, to see whether it matches the combined result, then a held-out check of the combined edit, and maybe a higher dose later.

I'm going to check the brake-only variant next, dropping the A edits and keeping just the ×2 down_proj change, then test ×3 to see whether over-suppression causes side effects.

Now I'm saving a copy of the current best edit, the ×2 version with the A edit, before I run anything else. I need to check what's in /work/edited and how the earlier best copy was stored.

