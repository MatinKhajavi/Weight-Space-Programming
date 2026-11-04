The copied token's total direct logit attribution creeps up only slightly, from about 19.3 to about 20, a gain of roughly 0.6 to 0.7 logits. That looks small in raw terms, but those raw numbers carry a shared component that moves every token together. The logprob moves from about -0.2 to -0.03, which is really a statement about the margin against competitors, so I need to check whether those competitors are dropping too.

The early growth in the copied token's logit is mostly coming from the L23 mixer, and the even cycle 0 already sits near 19 with a high logprob, so this looks like a near-fixed point where the content was already predictable rather than a strong self-reinforcing loop. The remaining loops don't seem to hinge on one component. I'm weighing whether to stop exploring and spend the remaining passes on held-out confirmation runs.

I'm weighing whether to keep exploring or spend the remaining passes on held-out confirmation. One option is checking whether the copied-token edit damages ordinary reasoning text, by comparing teacher-forced logprobs on reference solutions between the original model and the edited one. If the edit leaves those logprobs intact, that's useful evidence that the change is targeted.

I'm going to test whether the edit damages normal reasoning text: I'll build reference math solutions in a raw chat format with the solution inside the thinking block, then compare mean logprobs over those solution tokens between the original and edited models, looking at digit tokens separately.

I'm weighing a side-effect check against more held-out confirmation: running teacher-forced logprobs on reference math solutions for the original and edited models, to see whether the edit leaves ordinary reasoning text intact. I'm also looking at whether the GSM misreadings and the answer formatting (dollar signs, commas) matter, though those look hard to fix.

I'm checking the answer length distribution next, comparing token-count quantiles for the correct, wrong, and cut-off answers from the edited generation run to see whether verbosity is what separates them. I'm also weighing a possible "doubt" brake in the later layers that would suppress words like "Maybe" and "Perhaps", though the earlier attempt on that didn't change anything.

I'm checking whether a stronger cut in line-initial "Maybe" would help. The earlier variant that cut it by about 36% changed accuracy by only 0.3 points and left the cut-off rate at 27.3% against 28.9% for the baseline, so the gain looks negligible. I also want to see whether the final answers of correct versus wrong runs differ in how they're extracted.

I'm weighing whether a side-effect check on reference math solutions is worth the remaining passes, since it would show whether the edit leaves normal reasoning text intact. Another option is to look at what happens right before the loops start in the GSM cut cases, which are about 14% of GSM and would be the largest gain if they could be halved.

I'm checking the text right before the loop begins in the GSM cut cases, looking for what changes in the strawberry problem's output. I'm also considering whether the loop starts with a nested list of interpretations that keep repeating the same line.

I'm pulling up the excerpts now to see what happens right before the loop begins.

