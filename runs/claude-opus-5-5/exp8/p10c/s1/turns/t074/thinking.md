I'm checking how many of the leftover loops are generic repetition versus self-questioning, and the self-questioning ones look like a second mechanism, separate from the hedge-word loops.

I'm now planning to test damping the copying heads, which means choosing which heads to target and how strongly. Several layer 15, 19, 23 heads show high attention to congruent positions during loops, but that attention mass alone doesn't show they copy token identity, so I'd need to check their value-side output as well, since damping them could also hurt legitimate copying of numbers from the problem.

I'm weighing whether to test the head damping on a fresh set of problems or reuse the same set, since reusing it lets me compare against the earlier result directly but the two passes would regenerate the original side with different randomness. Using a fresh set avoids that cross-pass noise and keeps the selection of targets from biasing the comparison.

I'm weighing whether a sustained pair of edits could be compared directly, but the tool only runs the unedited model against the current edits, so each variant has to be measured against the original on the same problems. The original samples then act as a shared anchor, which means the effect of the new heads would be judged against the earlier +25 result rather than against the other edit itself.

I'm weighing which second mechanism to target, and damping the copying heads looks like the most direct fit for generic lock-in, though it carries a risk of hurting legitimate number copying from the problem. The other options, boosting doubt-promoting neurons or pushing the N998 effect further, seem less promising given that most remaining loops contain no doubt words at all.

Now I'm working out which heads to examine: looking at each head's attention to earlier positions that match an n-gram in the prior text, and separately how much attention goes to the problem statement, since heads focused on the prompt are likely the ones handling the math.

I'm going to score each of the 48 full-attention heads for how strongly its output value copies token identity, sampling about 2,000 common tokens so the calculation stays tractable. That tells me which heads are actually writing the repeated token, not just attending to matching positions.

I'm now picking which heads to damp, filtering for ones that both attend heavily to congruent positions in loops and actually write the repeated token's identity. I'm also noting that the gate on each head is driven by the current position's residual, so it doesn't offer an easy loop-specific switch. The linear attention layers could contribute to copying too, but the full-attention heads with the strongest congruent attention remain the main suspects. I'm leaning toward scaling those heads' output projections by about half, though I'm still considering whether 0.3 would give a clearer dose-response.

I'm about to run the copying score analysis to see which heads write the repeated token's identity.

