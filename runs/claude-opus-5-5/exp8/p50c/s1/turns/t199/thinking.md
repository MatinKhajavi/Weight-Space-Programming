I'm checking which loop line-start tokens carry the most weight. Token 23.6 looks general, since it's similar in pre-loop and within-line contexts, while 23.4 stands out with an excess of about +0.66 over pre-loop starts. Nothing points to one specific line-start copier, though 23.4 keeps showing up.

So 23.4 looks like the main head-level lever, but halving it would likely strip about 0.85 logits from correct predictions in normal text, which probably hurts more than it helps. Ablating it in B cost about 14 points, mostly from 23.4 I suspect, so I'm less inclined to test that. Maybe the bigger issue lies elsewhere: the remaining loops (82 of 512, plus 44 long non-loops), or the presence penalty pushing the model toward new tokens in long texts.

I'm now wondering whether the "Maybe" doubt-spiral pattern is driven by a single neuron. The idea is to check the layers 16–23 down-projection columns for ones that strongly promote "Maybe," then compare their activations at line starts before "Maybe" against other line starts in the cut windows, since there are few clean "Maybe" line starts to work from.

I'm going to check the fwd_11 windows for line-initial "Maybe" or "maybe" tokens, since those doubt-spiral windows are where that pattern would show up. I'm also weighing whether suppressing "Maybe" would even help, given that the earlier drop in its rate didn't move accuracy.

I'm weighing whether it's worth testing a third dose of the brake, but the expected gain looks small and the risk of harm looks small too, so I'll put it aside. Instead I want to check whether the cut texts that loop after the final answer, past </think>, show a different failure mode, and how many of them actually contain that closing tag.

I'm going to check whether any of the cut texts actually contain a boxed answer before the loop starts, since that determines whether they can still be graded.

I'm going to count how many of the cut texts include a "\boxed{" anywhere, since that would show whether the tool actually pulls out an answer from them.

