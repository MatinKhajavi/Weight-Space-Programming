Balanced accuracy on the remainder is 56.05% against C's 60.7%, a drop of 4.7 points, with the cut at 35.9% versus 24.6%. That's below C but not as far down as the A level on S1 (35.2%). I don't have a direct measurement of heads-only A on SH512, and I'm still weighing how S1, which is half of SH512 at 256 problems, compares with it.

So A on SH512 is probably around 35 to 40 percent. The remainder-doubled condition (BR) reaches 56.1, far above that, which suggests the non-Wait part of the column helps too. I'm wondering whether it suppresses Wait-like tokens outside the 2D span, or works indirectly through L23.

I'm checking whether the remainder suppresses Wait indirectly, by comparing line-initial "Wait" rates per 1k lines across BR, C, and A. If BR's rate is low like C's, the remainder works through L23 or correlated directions; if it's high like A's, it helps by another route. I have nine Python runs left, so I'll spend one on this analysis.

I'm checking which generation files correspond to the A and C conditions on S1, then planning a paired comparison of BR against C along with a cross-tab of the line-initial starters.

I'm checking whether the remainder suppresses Wait-like doubt tokens beyond "Wait" itself, since r is large enough to shift many logits by small amounts. I'll compute its effect on candidate starters like "Hmm", "But", "Actually", and "Alternatively" to see if it acts on those too.

The next part of the plan is to check whether the remainder suppresses other doubt-type starters, such as "Hmm", "But", or "Maybe", on top of "Wait". I'd compare the logit effects of the full column, its Wait-direction component, and the remainder on those candidate tokens. If the remainder pushes those down too, the brake looks like a broader doubt-family suppressor, with the Wait component just being the largest piece.

I'm weighing whether the remainder also pushes down other doubt-type starters like "Hmm," "But," or "Maybe," not just "Wait." If it does, the brake looks like a broader doubt-family suppressor and the Wait component is just its largest piece. I'll also restore C as the working edit, since BW reached the same score and C has been validated on more sets.

I'm checking whether the remainder also reduces Wait indirectly, so I'll compare line-initial Wait rates across the conditions on S1. I'll also check whether the remainder pushes down other doubt-type starters like "Hmm", "But", or "Maybe".

I'm pulling the S1 comparison together: C, BW, BR, A, and the original, each restricted to the S1 ids in /work/S1_ids.json. I'll compute accuracy on that subset for every condition, and I'm leaving out the brake-only run since I don't know which generation file it corresponds to.

Writing the code to compare line-initial starter rates and logit effects of the brake components on doubt-type tokens.

