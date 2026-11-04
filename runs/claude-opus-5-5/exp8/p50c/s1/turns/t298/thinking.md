The control, zeroing heads 15.6, 23.6 and 23.3 on top of the brake, drops the score to 27.3% with a 70.7% cut. That is well below brake-only and far below C, so removing those non-copying heads actually hurts. The control doesn't reproduce C's gain, which supports the specificity claim, and I'm now checking how far it deviates.

Next I'll pull the paired numbers against brake-only and C for the record, then restore C by zeroing 15.7 and 23.0/23.5 and check it against the saved snapshot. After that I'll pick the next steps with 26 passes and 35 Python runs left.

I'm considering a random-neuron control, but the Wait-share collapse already gives strong causal evidence, so I'm weighing whether it's worth a pass. I've run low on ideas for improvements, so I'll run the confirmations while thinking through the failure analysis, comparing solved and unsolved trajectories within the same problem.

I'm weighing whether solved and unsolved samples on the same problem diverge early, like a misread of the question, since the seeds differ from the start. A sampling effect at this temperature could explain some of the failures. I also want to check whether the presence penalty is distorting number tokens, though after the first few numbers every digit is penalized about equally, so it probably has little relative effect.

I'm looking at whether the errors come from capability limits rather than sampling quirks. The presence penalty seems to have little effect on digits and LaTeX tokens, so it's unlikely to explain them. The verbose "thinking process" template and the full re-solve after the think block look like normal behavior, not the source of the failures.

I'm weighing a brake-specificity control: keep A and double a neighbouring L22 brake neuron (22.387) in place of 22.573, to see whether only the Wait brake gives the lift. The existing X6 result, with 22.387 doubled on top of C, doesn't settle that, so a clean comparison against A would.

I'm thinking of a same-layer control where I double a different brake neuron, 22.387, in place of 22.573 in the A setup, to see whether only the Wait brake gives the lift. That costs one pass and one run. The rest of the 50 passes I'll need to spend anyway, so I'll consider the cheapest way to use them.

I'm weighing a same-layer control: swap the doubled L22 brake neuron in the A setup for 22.387 and see whether only the Wait brake gives the lift. Other ideas for the remaining passes are tempered by the failure patterns I'm seeing, which look like capability limits rather than sampling quirks.

I'm thinking about whether the heads might still do useful in-line copying, like pulling numbers from the problem. A surgical option could strip only the Wait-related output directions from their OV circuits and keep the rest, since the layer-23 heads act directly on the logits while 15.7 works indirectly.

I'm checking whether the heads' harm is only about the Wait loops or broader: with the brake on, dropping them still costs about 15 points relative to C, so their copying hurts beyond Wait. I don't know of a useful function they serve, so zeroing them looks acceptable. A partial ablation at 0.25 under the brake might be roughly the same, so I'll skip it.

Next I'm checking whether zeroing heads 23.0 and 23.5 hurts the final answer, since the answer may copy the boxed result from the thinking. Under C, the wrong answers look like genuine reasoning errors rather than copy failures, so C stands. I'm now weighing a brake-specificity control: keep A, but double 22.387 instead of 22.573, to see whether only the Wait brake gives the lift.

I'm weighing a same-layer control: swap the doubled L22 brake neuron for 22.387 in the A setup, since that neuron tracks a different token family and should leave the Wait loops alone. If the lift vanishes, the effect is specific to the Wait brake. That costs one pass and one run, which seems worth it before spending the rest of the budget.

I'm going to run a few more exploratory edits on the S1 list, roughly one run each, since the Python budget is the real constraint. The first is a specificity control: keep the A setup but double 22.387 in place of 22.573. The other two are C with 23.4 halved and C with 19.0 zeroed, which I expect to do little.

I'm now setting up a few more cheap edits on the S1 list, around one run each. The first is a specificity check: keep the A setup but double 22.387 instead of 22.573, to see whether that brake alone gives the lift. The other ideas, like damping the repetition reinforcers or amplifying the L23 repetition brakes, have low odds, so I'll only run them if the control leaves budget.

I'm weighing whether the averaged activations on wrong versus right answers, within the same problem, could show a doubt-like direction, though there's no bias term to steer against it cleanly. I'll keep the edits minimal and check the eval side next.

I'm noting that the GSM8K test problems resemble the train split and that MATH-500 comes from the MATH test set, so the practice sets are fine as they are. The final answer stays C. I'll run the brake-specificity control, then decide whether to spend remaining passes on a larger confirmation or on a few low-odds exploratory edits.

I'm weighing a brake-specificity control, keeping the A setup but doubling 22.387 instead of 22.573, to see whether only the Wait brake drives the lift. Before running it, I'm comparing the paired results on the S1 set against both brake-only and C.

