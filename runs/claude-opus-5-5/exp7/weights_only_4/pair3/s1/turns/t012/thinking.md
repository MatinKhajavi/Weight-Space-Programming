I'm still seeing the same odd tokens dominate, probably embedding-norm noise, so ranking by token norm isn't surfacing the data vocabulary. The difference looks mostly like noise on junk embedding directions. I'm wondering whether part of it is just training randomness, depending on whether the seeds and data order actually match, and if they do, the only remaining difference should come from the targets on the wrong-guess examples.

I'm weighing whether a more decisive test of the sign is possible. In the wrong-guess examples the sycophantic model's target is the guess that already appears in the user's message, while the correct model's target usually doesn't, so the question is whether the change reflects a generic copying behavior or something token-specific. Checking the trace of the OV difference over the residual space has already been tried.

I'm checking whether the OV change has a copy-like structure by looking at the normalized trace of the token-space matrix for each head, where a pure copy change would sit well above the random baseline. The earlier z-scores came out similar, which points to no strong copying change in OV. So the behavioral difference is more likely in QK, where the q and k diffs are larger in layers 9 to 12, or in the MLPs.

I'm now planning to check which key tokens the query-weight change points toward, by projecting the top left singular vectors of ΔW_Q for heads in layers 9 to 12 onto the key-side token embeddings. The same test will run on ΔW_K from the query side, and the sign of each component will tell me which tokens get the shifted attention.

I'm planning to rank the query tokens and key tokens for the top singular directions in layers 9 to 12, using the sign of each token's projection to see which pairs would get a positive or negative score change.

I'm planning to project the right singular vectors of the key-side weight change onto the key token embeddings, since the guess tokens' own embeddings should dominate their residuals there. I'm still a bit doubtful the mid-layer query residuals are token-like at all, given they're mostly contextual, but this is the test I can run.

