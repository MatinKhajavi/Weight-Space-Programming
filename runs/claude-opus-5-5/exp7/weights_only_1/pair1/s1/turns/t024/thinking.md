The null distribution looks too narrow here, since the z-scores run as high as ±7 and the head-level values often exceed 1 in magnitude, which suggests the structure isn't being captured. Early layers show a combined positive score, but the calibration seems off. Several heads also show strong positive matches on duplicate tokens, such as L0h1 and L3h3, which points to duplicate-token behavior.

Now I'm checking which heads show strong token-matching signals, both positive ones like L0h1, L8h0, and L3h3 and negative ones like L7h2 and L7h3. The negative heads and the positive heads pull in different directions, so they may not be doing the same thing. I'm also wondering whether the correct model needs token-matching for a different reason, such as restating parts of the question in the response.

I'm stepping back to weigh the remaining budget of 37 runs against the direction evidence, which so far only points weakly toward A at about 2σ. Copy-suppression seems worth checking next: if the correct model learns to ignore the guess rather than suppress it, that would show up as a distinct pattern in the heads.

Copy suppression seems to matter here. In the base model, some heads like L7h1, L8h0, L9h1, and L15h1 push against copying tokens that are being predicted. If the syc model weakens that suppression, the guess gets copied more readily, while the correct model needs to keep suppressing it when the guess is wrong.

I'm checking whether the copy-suppression heads attend to the guess at all, since their O-term signs don't line up cleanly with the direction evidence. If they don't attend to the guess, the output-side sign wouldn't tell me much, so I'm now looking at how the query side reads the predicted token.

I'm weighing whether the copy-suppression heads attend to the guess at all: if the correct model's residual points at its own prediction, those heads would ignore the guess, but if the model leans toward the guess, they'd suppress it, which helps on wrong guesses and hurts on right ones.

Checking the token-matching scores for the copy-suppression heads now: all four (L7h1, L8h0, L9h1, L15h1) got more positive in the correct-vs-syc comparison. If copy suppression works by attending to tokens that match the prediction, stronger matching would suppress the guess, which points toward A being the correct model. That conflicts with the output-side sign I got earlier, so I still need to check the copy heads too.

I'm looking at how the copy-suppression heads' token-matching shifts across the remaining layers, comparing the correct and syc models.

The copy-suppression heads all picked up more token-matching under A, while the strong copy heads mostly lost it, so the pattern points toward A being the correct model, which conflicts with the earlier output-side sign.

The copy-suppression heads all gained token-matching under A, and the strong copy heads mostly lost it, which would make A look like the correct model. That runs against the weak positive output-side signal, so I'll need to check how the query side reads the prediction before settling on a direction.

I'm checking whether the copy-suppression and copy heads really move in opposite directions under A, since that would flip which model looks correct. The token-matching shifts seem to run against the earlier output-side sign, so I'll compare the two signals across heads before settling on a direction.

I'm checking the QK token-match pattern across the late heads, correlating each head's base copy score against its change in matching, since that would show whether the shift runs with or against copying. I also want a single signed sum of copy score times matching change, so I can see the net direction in one number.

I'm checking whether the base match scores for the copy heads line up with their copy scores, since a generic shrinkage of heads with strong matching could produce the anti-correlation I'm seeing rather than a real behavioral difference.

I'm breaking down each head's contribution by its base copy score, base token match, and how much that match shifts once the gain is separated from the perpendicular change.

