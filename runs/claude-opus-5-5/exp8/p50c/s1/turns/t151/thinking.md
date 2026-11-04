For the GSM problems, I'm breaking the failure modes down: the correct answers show up around 13% into the text, while the wrong ones and the cut-off ones are where the correct value surfaces much more often than the control rate, and the cut-off ones show it very early, at about 4%, before the model spirals on for tens of thousands of tokens.

Looking at the failures, most of the wrong and cut-off GSM runs had the right answer early and then doubted it. So the real problem seems to be not committing after solving, rather than failing to compute.

I'm checking how the correct texts end their thinking, specifically what comes right before the closing tag and whether they use commit phrases like "Final Answer". Then I want to see whether the failing texts ever reach that kind of commit statement, since earlier notes suggest they mostly keep doubting instead.

I'm looking at how a verified answer should lead to concluding rather than doubting again, since in the failing runs each check seems to add a new doubt. Finding the direction that separates "about to conclude" from "about to doubt again" would be a cognitive-level fix, but a constant push to conclude earlier regardless of the problem would be off the table.

I'm weighing whether a steering vector that pushes toward concluding would count as closing early regardless of the problem, which would break the rules, so I'm leaning toward focusing on repetition and on a specific doubt component like the Wait brake.

I'm checking which neurons push the doubt tokens at line starts in the pre-loop region, starting with whether the Wait promoter also boosts words like "Maybe", "Hmm", or "Actually". My guess is that cutting it only swaps one doubt word for another, so I'd look at activations at position 581 where " Let" competes with " Is".

I'm weighing whether to look at the neurons that push doubt-starting tokens like "Let's check" or "Maybe" against the ones that push a conclusion like "So" or "Therefore," since "So" can open either a conclusion or another doubt. Comparing internal states at line starts in the correct texts just before their final answer against the failing spirals would be correlational, though, so I'm not treating it as a direct mechanism yet.

I'm still circling back to the loops, since breaking them let the model reach its answer in some cases, so they still matter. In the W1 runs the driver looks like layer 21's mixer together with the layer 19 and 20 mixers, while the original loops pointed at layers 15, 19 and 23 instead.

I'm now checking whether the L21 push toward the copy token shows up in ordinary line starts as well, or only inside the repeated loop lines. The comparison is between repeated lines and lines that reuse a first word like "If" but say something different.

I'm now checking which layers push the copy token in the early repeated lines, and whether the same pattern holds across the whole line instead of just the first content word. I'm also weighing a cheaper route: testing candidate edits by generation accuracy, using the attribution results to pick them.

I'm weighing a causal test instead: zeroing the output columns of a group of L21 DeltaNet heads and re-running on the two prompts to see how p(copy) shifts at the decision points, using binary splitting to isolate a dominant head.

I'm weighing a single-pass check: zeroing a set of L21 heads and comparing the mixer's contribution to the copy token with and without them, since layers before 21 stay unchanged and the prefix is identical, so the difference isolates those heads' share.

I'm planning to split the layer 21 mixer's contribution into per-head pieces by zeroing groups of heads in each pass, since the inputs to layer 21 stay the same and the mixer output is linear in the heads.

I'm splitting the 16 layer-21 heads into groups and ablating each group in its own pass, since the mixer output is linear in the heads and the inputs to layer 21 stay identical, so the difference isolates each group's share. One pass per head is exact but costly, so I'm looking at adaptive splitting to find the dominant heads first.

I'm weighing whether to split the layer 21 heads into groups with adaptive passes, since if one head dominates, four ablation passes would isolate it. Before committing to that, I'm checking whether the per-head contributions can be read directly from the weights and returned states, since the mixer output is a linear map over the head outputs.

I'm checking whether ablating a DeltaNet head is worth doing at all, since it might also hurt the in-context copying the model needs for numbers from the problem. The earlier attention-head ablations hurt more than the A-set heads did, so I'm weighing whether a head-level cut would have a similar broad cost.

I'm weighing whether a direct generation test or a cheaper look at L19 attention would tell me more than ablating a single L21 head. Zeroing a DeltaNet head is hard to target without knowing which one, and the earlier L19 ablations already hurt, so I'd check the attention patterns at the decision positions first.

I'm checking which attention heads in the full-attention layers still do induction-style copying at the decision points. For each head I'm measuring how much attention lands on the token that followed the previous occurrence of the current token, comparing lines inside the loop with lines before it.

