Since the losses never got past a lead of 20.5, the fade at 25 couldn't have been what changed them, so the lower score looks like noise or ordinary variance. I'll keep the fade anyway, since it guards against weak play. White's early lead of 8.5 to 12.5 in those games is just the 7.5 komi showing up, which is fine. Overall I'm putting the current state at roughly 0.6 to 0.7.

I'm weighing the remaining passes and runs now, noting that the losses against the strong bot look like mid-game issues and Black still trails White. The most promising next step seems to be the opening, where tengen should be favored as Black, so I'll consider the opening bias alongside the off-track parameters and the contact setting before settling on a few 128-game experiments.

I'm checking whether the escape and capture values are too weak against the edge penalties, since a 1st-line escape scores lower than a quiet 3rd-line move and might get abandoned. The rescaling of the positional terms without matching changes to escape and capture could be why those tactical moves are underweighted.

I'm going to test an escape bonus of 14 instead of 9, keeping the capture value at 16, since edge escapes currently look underweighted next to the positional terms.

