I'm tracing the OV-circuit for each head using raw unembedding vectors as a stand-in for the residual stream. Head 23.2 comes out strongly suppressive (z -5.0, top rank), attending heavily to itself and to duplicate tokens, so it looks like an anti-stutter mechanism that pushes down repeats of the current token. Head 23.6 is only mildly suppressive, and I'm skimming the remaining heads next.

I'm checking the copy heads 23.0, 23.4, and 23.5 next. They show near-zero scores with this proxy, which likely means the raw unembedding doesn't capture how they read from the residual stream.

I'm weighing whether amplifying 23.6 would matter much, since its suppression is weak compared with 23.2's. The ablation hurt a lot, but I can't tell yet whether 23.3 or 15.6 is responsible. Next I'm looking at 23.2, which suppresses strongly by attending to itself and to duplicate tokens, though it doesn't look like a loop brake.

I'm checking whether 23.2's suppression, which runs through its duplicate-token attention, might work as anti-repetition on the continuations that followed earlier copies of the current token. If so, amplifying it could help, though I'm not sure its suppression holds up against the copy heads in loops.

I'm weighing whether to test amplifying 23.2 at 1.5x alongside 23.6, or a combined setup, since each run is costly and there's no strong prior. Its roughly 0.5 to 1 logit of suppression probably won't break deterministic loops, though it might reduce how often loops form.

I'm considering whether amplifying 23.2 could hurt numbers with repeated digits, since Qwen tokenizes digits individually and its anti-doubling suppression might push against "100" or similar. That would make the test risky, so I'd want to check it before committing.

I'm weighing whether amplifying 23.6 at 1.5x is worth testing, since its suppression is mild and might only nudge numbers with repeated digits. The control result, where removing 15.6, 23.6, and 23.3 lifted cuts from 54% to 71%, suggests one of them matters a lot, though I can't yet tell which.

I'm weighing whether to test amplifying 23.6 at 2x or 1.5x. A 2x run should give a clearer dose signal, though a larger multiplier also risks more damage, so I'm leaning toward 2x and would follow up with 1.5x and 3x if it helps.

I'm considering whether to first run the 23.6 ablation as a diagnostic, since it would show whether that head is the key anti-loop component before I commit to amplifying it. That costs one pass, which the budget can absorb, and it would give the low end of the dose curve.

I'm going to run the 23.6 amplification at 1.5x now, since the edit is already prepared and that saves a setup step. After that I'll pick the next test, either the ablation of 23.6 or a higher multiplier, depending on the result.

