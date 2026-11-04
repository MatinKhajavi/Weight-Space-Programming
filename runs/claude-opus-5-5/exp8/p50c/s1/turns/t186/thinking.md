Looking at the paired comparison now: the V2 versus C difference is about +0.39 with a standard error of 2.32, and the switches run 18 one way and 17 the other, so there's no real effect. The "Ready" rate sat at 0.01 per 1k tokens, which makes sense since those lines were rare to begin with. I'm also noticing that "Maybe" dropped from 2.68 to 1.71, and I'm checking "Okay" next.

Going with C for now, since V2 gives no measurable gain and is more complex. Even small edits seem to shuffle a lot of individual answers, so detecting effects of about 3 points would take far more data than is practical. I should look for bigger levers, around 5 points or more, and the ones that have worked so far are the induction head ablation and the Wait brake.

I'm weighing a few other possible levers: the other induction-mass heads that the B ablation hurt, whether the DeltaNet decay rates could be what handles long-range copying, and why the loops only start around 16k tokens, maybe through attention in the full-attention layers drifting at that length.

Going through the presence penalty point now: with it set at 1.5, every token already used gets pushed down, which could nudge the model toward unusual wording or drift in long reasoning, though the digits would all be penalized equally once they've appeared.

I'm going to look more closely at the C cut texts, sampling excerpts at 25%, 50%, 75%, and the end, to see what the model is doing in the non-looping cases and at the start of the loops. I want to find out whether something like growing indentation, bullet hierarchies, or repeated re-reading of the problem is driving it.

I'm now checking whether the wrong answers are really reasoning errors or just formatting mismatches, like a boxed answer with units that the grader rejects. If a lot of them are formatting problems, that's a cheaper fix than changing the model's reasoning.

