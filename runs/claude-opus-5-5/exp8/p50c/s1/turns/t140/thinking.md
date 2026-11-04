I'm settling on inline text instead, since there's no file option. I'll build a handful of short windows with ASCII-only content so they can be pasted directly.

I'm planning compact windows of about 150 tokens before the onset and roughly 350 in the loop, keeping indentation modest so the characters stay easy to reproduce exactly.

I'm weighing whether dumping the full request as JSON is worth the output cost, since I'd still have to retype it all. Before doing that, I'd rather settle which mechanism to test, and the 11.1 induction head looks like the most promising candidate.

I'm going to measure, for each head, how much attention goes to the previous cycle in the loop versus before it. That should show which heads concentrate on the repeated pattern, and I can test ablating the strongest ones. I also want the direct logit attribution by layer, which needs hidden states from the same pass. Two windows of roughly 1500 tokens each should fit inline.

I'm sizing the two windows now, one GSM bullet loop with short indentation and one MATH bullet loop, and I'm aiming for about 700 tokens before the onset so the lines stay easy to reproduce exactly.

Checking the attention pattern on a couple of bullet loops next, since the tokens need to be ASCII-safe and the indentation kept modest so they can be reproduced exactly.

Picking two windows to test: a GSM8K strawberry bullet loop and a MATH number theory loop with a one-line repetition followed by a multi-line cycle.

I'm picking the two test windows: the GSM8K strawberry bullet loop and the MATH number theory loop, each with a prompt prefix, some pre-onset context, and the start of the loop.

