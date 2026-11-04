I'm reading the loop-token numbers: the edit lowers lock-in at the k5 and E6 positions, though the average across all loop tokens is still pulled up by copied within-line content, which is expected since the edit only acts at decision points. Next I'm pulling out the per-period escape probabilities under sampling, comparing the original's drop-off at k3 through k5 with the edited version's flatter values.

I'm checking the escape probabilities under E6, which stay at 85% or higher from k3 on, versus under 7% for the original at k≥4, a strong sign the edit works as intended. Next I want to see whether the roughly 3% of lead tokens with large logprob shifts are mostly the targeted opener tokens, which would confirm the edit is specific to that decision.

I'm working out why the token count doesn't line up: 3730 rows against 4802 scored tokens, and the body rows come up about 63 short of what the per-text lengths predict. I'm weighing whether the last partial period or boundary tokens are being dropped, and whether the analysis is pulling per-position period labels for each text.

I'm looking at which line-start tokens carry the biggest logprob shifts, checking whether the "Wait" positions specifically reflect the targeted edit. To pin that down, I need the token ids for "Wait" and " Wait", and I'll pull the line-start tokens from the body and check which ones match.

Next I'm deciding which tokens to examine: P(actual opener) at each line-start decision point by period, for both the original and E6, plus P(Wait) on its own, then checking what the lead's large logprob shifts consist of.

