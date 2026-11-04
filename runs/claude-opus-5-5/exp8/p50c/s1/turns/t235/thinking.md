The tiny push values come from leaving out the division by the final rms, which is about 0.28. Multiplying by roughly 3.6 only brings the largest value to about 0.14. The neurons' effect on the doubt-versus-progress direction still looks small, and no single neuron stands out as a strong doubt promoter at line starts once the averaging is taken into account. I'm also checking how that averaging over the 26 doubt tokens affects the result.

The doubt push seems to come from attention in layer 19 rather than from any single neuron. I've put a lot into the doubt and loop direction, so I'm stepping back to decide how to spend the remaining 33 passes and 90 Python runs. The C edit is already strong and robust, so any further change has to show a clear gain. The biggest candidates are the MATH L3-5 failures, which look like they come from looping or long exploration rather than something fixable, and whether the edits are costing some verification ability that could be restored.

I'm weighing a softer version of the head ablation, at roughly 0.25 strength, under the brake, since copy-head pressure might be lower there and partially restoring copying could cut the wrong answers without adding much looping. The existing dose numbers suggest the optimum sits near zero, so the expected gain is small, maybe 0 to 2 points on one pass over 512 items.

I'm looking at whether a softer head ablation, around 0.25 strength, could help under the brake, since copy-head pressure may be lower there. The earlier dose numbers put the optimum near zero, so I expect only a small gain of maybe 0 to 2 points on one pass over 512 items. I'm also weighing a more targeted Wait-only brake that avoids side effects on Hmm, Ready, Actually, and </think>, though gen_21 showed no gain.

Still weighing whether the "factor of 2" wrong answers on GSM could be a separate class worth looking at, since the short in-line stutters like repeated "Half dozen = 6" look like plain short-period copying and probably share the same heads, so fixing them without side effects seems unlikely.

I'm wondering whether the factor-of-two errors on the GSM items reflect a systematic bias in how the model reads "each" versus "total." I'll look at a few of them, such as the 1880, 4694, and 1699 cases, to check whether they share a misreading like treating a round trip or pairs as doubling.

Next I'm going to look at a few of the factor-of-two GSM problems, specifically the problem text and where the model decides on its answer, to see whether there's a shared misreading.

