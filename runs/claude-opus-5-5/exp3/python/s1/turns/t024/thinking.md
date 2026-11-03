Looking at the singular value spectrum of the stacked OV matrices, it decays gradually without a sharp cutoff until flattening out around the high 20s, suggesting an effective rank somewhere near 27-28.

I try other distinguishing angles — the "space-space" and "newline-space" attention patterns, positional range of attention for Python indentation versus English induction — but none give a clean separating signature between the two cases. I consider whether broad document-level statistic heads might differ (topic words vs identifier vocab), and whether the L0H4 positional-average signal has any structural reason for needing absolute position.

I settle on newline frequency as a more promising lead: English prose windows rarely contain a newline at all, while Python windows have newlines roughly 3% of the time, much more common than in English text, which could be a usable statistical difference.

Rare letters like j, q, x, z also stand out—English has about five weak output directions when including newline, while Python has only three (minus x, which is common in Python due to "x", "max", "index"). This split roughly matches the observed pattern of 3 weak directions and 24 strong ones, weakly favoring Python. I check the singular value gaps more carefully: the ratios within the strong group stay close to 0.93–0.99, while there's a sharper drop after position 24, reinforcing that there's a real boundary there separating the strong and weak groups.

For English I'm estimating maybe 4-5 weak letters once accounting for rare characters plus short lines inflating newline frequency. For romaji I'm weighing how common p, f, j actually are versus truly rare letters like v, l, q, x that barely appear at all.

Counting frequent tokens per language: romaji has about 23-24 strong dimensions (letters plus space and newline), with v as a borderline rare case making the predicted strong count close to but slightly off from what's observed. Python matches exactly at 24 strong plus 3 weak tokens out of 28 total, while English has 28 tokens with roughly 4 rare ones.

This near-match across languages isn't decisive on its own, but it builds a consistent case for the Python hypothesis. I'm also checking whether direction 0 could itself represent a token distinction (like space vs newline) — if so, the counts would overflow past the 27-dimension maximum imposed by softmax shift invariance after the 28-token final layer norm, which would be impossible, so that direction must be something else entirely.

So in romaji only ~23-24 tokens appear meaningfully as targets since l, q, x are rarely learned targets, shrinking the effective subspace further. Yet L3H1's copy head still writes roughly uniform ~2.0 weight into directions 3-27, including 25-27 — which would only make sense if those directions correspond to ultra-rare tokens actually present in context for copying.

If python, q/z/j repeat in identifiers like "json" or "query" and get copied naturally; if English, those letters rarely repeat enough to be copy targets. I'm leaning slightly toward python over English, maybe 55:45, so I want to look for more discriminating evidence — specifically the L2H2 "bigram-matching" head that attends to earlier positions where the previous token matches, letting it predict based on repeated bigram patterns.

Checking the K-composition numbers for L2H2 into L3H5, I see only a weak signal above baseline. Let me pivot to thinking about Python's indentation structure—maybe some head specifically attends to the most recent newline, which would be valuable for tracking line position in Python but less relevant in English text. I want to check key-side token preferences across layers 1-3 to see if any head shows a strong bias toward the newline token.

