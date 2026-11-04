I'm checking the q_proj shape: 4096 by 1024 splits into 8 heads of 512, each holding a 256-wide query and a 256-wide gate, so the layout looks like interleaved per-head pairs. Next I'm looking at how the projection s on feature 22.573 tracks activation act573, which is only weakly correlated at 0.18.

I'm looking at the overlap in s between strong-activation positions and the rest: they separate only moderately, since the brake output is mixed with other components along that direction. A gate shift proportional to s would close the gate at the strong positions but also push ordinary positions in both directions, so the readout isn't clean. I'm also weighing an LDA-style direction computed in closed form from the activations as a cleaner alternative.

I'm weighing whether a closed-form mean-difference direction would be a fair readout here, though with only 24 strong-activation positions against 1024 dimensions, overfitting is a real risk and I'd need many more texts. The gating idea also looks marginal: even a perfect fix might add only a few points, with maybe a 30% chance it works as intended, so the cost may not be worth it.

I'm weighing whether to test a brake dose of x3, which looks cheap but small in expected value. I'm also considering whether the brake's suppression of </think> at line starts could explain the commit loop, since at act -4 it would cost about 0.5 logits there in both the native and modified cases.

I'm working out whether the brake's suppression of </think> at line starts matters in the commit-loop cases, since it would cost about 1.0 logit there instead of 0.5. The earlier notes suggest it's irrelevant in typical loops, so the question is whether the few commit-loop texts make it worth a targeted fix that keeps the native component and drops the doubled part.

I'm now planning a version of the brake column with its </think> logit component removed, and I'm checking whether the brake also suppresses closing tokens like "Final" or "**" at line starts, since doubling it could delay commits.

I'm now checking how strongly the brake direction pushes the closing tokens "Final", "**", "Done", "Therefore", "Thus", "So", and "The", since doubling it might suppress those at new-thought line starts. I need to confirm whether each of those is a single token, including the spaced and bare forms, before I compute their z-scores.

I'm going to check the closing tokens, including both spaced and bare forms, since doubling the brake could suppress them at new-thought line starts. I'll compute the logit effect for each one and see whether any of them are strongly pushed down.

