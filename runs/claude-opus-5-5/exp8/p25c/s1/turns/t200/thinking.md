I'm going to wrap up with a final answer that walks through the path, the edit, the evidence, and why the change should help, in plain prose. The diagnosis is that the original model mostly fails by never finishing: most answers run into the token limit, and nearly all of those get stuck in exact repeating loops, though every answer that did finish was correct. I'm also noting that most looping answers had already produced the reference answer before the loop started. Next I want to pin down where the failure begins.

I'm tracing where the looping starts: the loop token's probability climbs with each repetition while the escape probability collapses. Attention from a few late-layer heads, mostly in layers 15, 19 and 23, is what boosts the repeated opener at each line start, which is how the copy ends up locking the loop in.

I'm now specifying the final edit: an orthogonal projection on the output of the attention layers 15, 19 and 23, scaled by a per-layer factor, built from the span of eight line-opener directions. The numbers I'm checking are the held-out replication results, moving from 12.0% to 43.0%.

I'm now checking the held-out replications: accuracy rose from 12.0% to 43.0% on the combined set, a paired gain of 31.0 points, with GSM8K and MATH both improving and the share of cut-off answers dropping from 89% to 48%. Teacher-forced checks at the loop decision points show the probability of repeating "Wait" falling sharply as repetitions build, while escape probability rises.

I'm checking the specificity results next: the edit barely moves normal reasoning tokens, with only about two dozen of roughly 1,700 changing by more than a nat, and most of those are the targeted openers. The direct-logit attribution shows the Wait margin flipping in loops, while in ordinary reasoning the layer 19 contribution stays near zero, so context-driven choices seem to be left intact.

I'm checking the remaining limitations: about half the held-out problems still get cut off, mostly from problem-specific oscillations and doubt bullets driven by the same copying mechanism, and adding more openers just shifts the loops to new ones. I'm also expecting the test sets to behave like the practice distribution, so gains should resemble the measured ones, with some uncertainty.

