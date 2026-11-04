I'm noticing that at line breaks, the indentation choice is almost certain in these loop windows, so the presence penalty wouldn't change those tokens. The real question is where the model decides to nest deeper, which seems to happen at new sub-bullets rather than at the newline itself.

I'm weighing whether the depth increase is a sampler artifact or just the model nesting sub-points, and leaning toward the latter. Next I'll step back and decide how to spend the remaining passes, with the current edit already giving a solid held-out gain.

I'm checking whether the cut rate on MATH varies by problem level, since that would show whether the model is failing on hard problems or stalling even on easy ones. If the cuts also show up on level 1-2 problems, that points to a pathology rather than plain difficulty.

