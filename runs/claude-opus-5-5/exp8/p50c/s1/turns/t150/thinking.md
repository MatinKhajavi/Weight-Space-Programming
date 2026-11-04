I'm noticing that the embedding-level OV copy score misses the late-layer heads 15.7, 23.0, and 23.5, which all score near zero. So this metric probably isn't capturing what those heads move. Layer 11 heads, by contrast, show strong copy scores on several of them.

I'm looking at the copy-suppression picture now: 23.2 has a strong negative score, so it may be an anti-repetition head, while 15.1 looks positively copy-like. The late-layer copy heads don't seem to copy token identity in the embedding basis, so this metric is unreliable for them. I want causal tests instead, like measuring the copy token's logprob at loop-entry decision points and checking collateral damage on correct solutions.

I'm weighing whether to paste all texts into one forward pass, which would run to about 100KB and be impractical, so I'm considering generation passes as the causal test instead. Those cost more time but avoid the pasting bottleneck and measure accuracy and cut share against the C baseline by seed. With roughly 36 passes left, I can afford maybe 15 to 20 candidate tests and keep some back for final confirmation.

I'm weighing how to localize the L21 mixer's contribution to the copy direction, since the output projection can't be cleanly split into per-head parts from the mixer output alone. Ranking heads by how well their output subspaces explain the logit-difference projection isn't conclusive, because the norm bound on each head's contribution is loose. So I'm leaning toward a causal approach with forward passes that ablate individual heads instead.

I'm weighing whether to localize the L21 effect by ablating heads in groups, since that's a few passes at minimum. Splitting the 16 heads into halves and then narrowing takes about six passes just to find one head, and that head might not be safe to remove anyway. So I'm also considering a more mechanistic intervention aimed at what drives the loops: lines becoming self-similar in doubt spirals.

I'm weighing whether a broad intervention could still help. Even fixing every exact loop would only lift accuracy a few points, and the wrong answers are longer and driven by second-guessing correct solutions, so I'm checking what the expected gain really is.

I'm checking whether the wrong GSM answers had the correct result show up early and then get abandoned under over-doubting. I'd compare how often the right answer appears before the model commits, for both the correct and wrong cases.

I'm wondering whether the commit point is the lever: if the model stops second-guessing once it reaches a plausible answer, maybe increasing the chance it writes "Final Answer" or closes the thinking block would cut the wrong, overlong outputs. I'd check first whether correct solutions tend to contain the right answer early, before the doubt starts, and whether the wrong ones abandon it.

I'm thinking about whether to check if the correct answer shows up early in the correct samples, and whether the wrong or cut ones drop it later. That would tell me whether the failure is over-doubting after a right answer. I'll pull the position of the first correct-answer value in each sample and compare it across outcomes.

