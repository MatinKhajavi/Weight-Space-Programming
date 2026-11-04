Correlations this high, around 0.8 to 0.9 across many neurons in every layer, point to a global effect rather than something specific. The looping portion probably activates very differently overall, since its short, repeated lines carry less varied content, so many neurons move together. That level of shared correlation suggests the per-sample mean is driven mostly by the loop fraction, which doesn't tell me enough. I'll need a per-token analysis to identify the specific neurons.

I'm planning a set of causal experiments on the looping transcripts: first finding which full-attention heads attend back to the previous occurrence of a line, then checking which MLP neurons respond differently on repeats, and tracing where the copied token's logit gets written. The plan after that is to scale down the candidate heads' output projections and rerun the same text to see whether the loop behavior changes.

I'm weighing whether breaking the loops would even help: the cut-off texts are already around 6k tokens of verification before a loop starts, so stopping the repetition might just lead to more "Wait" checking or a fresh loop. I want to see how many of those samples would have finished on their own, since the finished ones run up to about 9.5k tokens.

I'm weighing whether an attractor-style explanation fits: the loop might be the in-context copying pull winning over whatever pushes against repeating a line, and small models would be more prone to it. The presence penalty doesn't help much here since every token has already appeared. I'm also considering whether the copying heads could be dampened for long-range exact copies, and I want to look at how the Gated DeltaNet layers, which make up most of the stack, compute their gating before deciding where an intervention would go.

I'm working out how the Gated DeltaNet layers do associative recall: the key is built from the conv-mixed features of the last few tokens and the value from the current token, so a later query can pull back what was stored when that key last appeared.

I'm checking how a DeltaNet layer could do induction: the key could be built from the earlier tokens and the value from the current one, so a later query matching that key would bring back the next token's information, which means the conv can handle it.

I'm thinking about how the copying could be spread across many DeltaNet layers, not just the full-attention ones, since a decaying state could hold a 42-token-back match easily. The next step is per-token analysis on a looping text, with ablation edits, but each one costs a teacher-forced pass from a limited budget, so I need to pick them carefully.

I'm weighing a teacher-forced setup on one looping transcript, where the loop starts early enough that a short text covers several repetition periods. Generation passes are costly enough that dense per-token measurements are the better way to localize the copying behavior.

I'm going to pull up the samples with the shortest onset and their loop periods, then read through their text to pick one for the teacher-forced test.

