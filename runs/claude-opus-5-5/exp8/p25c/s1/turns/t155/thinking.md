E6-half accuracy came out at 14.1%, with 84.8% of the cut, which is barely above the original average of about 11.5%. So the dose-response looks steep: a half-strength edit gives almost no effect, while the full-strength setting jumps accuracy by about 31 points, and going higher stays at that level. That reads like a threshold.

Now I'm checking whether the dose-response is a real threshold: the effect appears to start somewhere between half and full strength. I want to see whether the reversal is strong enough to overcome compensation, and whether the full-strength setting sits safely on the plateau or right at its edge.

I'm weighing E6 at s=1 against E6s at s=1.5. Both sit on the plateau, but s=1 is closer to the steep knee between 0.5 and 1.0, so if the eval set shifts the threshold, it might fall off. E6s has a bit more margin, though it is a larger edit. The pooled difference is +0.78 ± 1.78, so the two are statistically equivalent, with E6s slightly ahead on two of the three sets.

Checking the dose-response more finely, I'm weighing E6 against E6s on robustness versus parsimony: E6s sits further from the knee between 0.5 and 1.0, while E6 is the smaller edit. Both are on the plateau and statistically equivalent, so the choice comes down to how much I trust the knee location to hold on the other eval sets.

I'm checking whether E6s's extra "no answer" finishes (10 versus 2 for E6) point to slight damage from the stronger edit, since the answer formatting may be getting disrupted at s=1.5.

I'm going to check the no-answer cases for E6s, since its extra unfinished responses could mean the stronger edit is disrupting answer formatting. If that's the cause, it would be a reason to prefer E6. I also have four passes left, so I could run s=0.75 on the held-out set to find where the threshold sits.

I'm weighing whether to spend a pass on locating the threshold at s=0.75 to help choose between E6 and E6s. If that point is already on the plateau, E6 at s=1 looks safe, and I'd have enough confidence without spending more of my remaining passes on confirmation.

I'm going to split the E6-half numbers out by problem set and compare them against the original and E6 runs, then check the no-answer cases for E6s to see whether its stronger edit is disrupting answer formatting.

