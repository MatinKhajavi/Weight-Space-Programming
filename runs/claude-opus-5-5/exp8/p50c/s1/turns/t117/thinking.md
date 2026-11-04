I'm checking the final-answer loops under the brake setting: 5 of 74 were cut off, and one of those has a predicted answer of 196, which I'm unsure the grader would still score. That's a small group, so it doesn't look like a major pattern. I'm also noting the brake column's effect on logits, where "Wait" rises by about 0.30 and "</think>" by about 0.04.

I'm weighing whether to tune the brake further: the </think> suppression it causes could be about 0.6 logits at line starts, which is moderate, and it only touches five final-answer loops, so I'm not prioritizing it. The bigger remaining issues are the exact loops, especially short in-line copying, and long reasoning that doesn't finish within the budget.

I'm weighing whether a held-out check is needed before choosing between the ×1.5 and ×2 settings, so I'm planning a paired run on a mixed set of MATH and GSM problems from the held-out pool. The idea is to compare the original, ×2, and edited versions on the same problems to get a three-way read.

I'm turning to the held-out check now: deciding between the ×1.5 and ×2 settings on combined S1 and H1 data (512 paired problems), and planning to run a "both" pass on H1 next, since the generation budget still has room. After that I'll look into the remaining loops by testing the texts under the brake setting.

