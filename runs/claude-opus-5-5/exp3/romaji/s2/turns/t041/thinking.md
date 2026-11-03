Looking at L0H3's QK matrix in the Pin plane, the eigenvalues give a rotation of about 53°, which implies a period of roughly 7 positions—so Pin might actually encode a higher-frequency positional signal rather than a slow absolute-position feature.

L0H0's Pin block has a much smaller rotation of about 16°, suggesting a more local attention pattern, possibly focused on nearby or same-position tokens. I'm trying to reconcile how heads that query and copy Pin via OV would use a periodic signal like this, which seems like an unusual mechanism.

Stepping back, I'm weighing the high-level question of English versus Python as the underlying language, with roughly equal prior probability after setting aside romaji. One distinguishing signal could be Python's indentation structure, which would require the model to count consecutive spaces since the last newline—implying a head that sharply attends back to the most recent newline for positional counting.

I'm also considering structural differences: Python's short, column-aligned lines versus English's more uniform token distribution, and specific bigram patterns like (space, space) or (newline, space) that are common in Python indentation but rare or nonexistent in English text. Trying to use the MLP0 neuron activity and singular value spectra to infer token frequency patterns isn't yielding a clear mapping either.

I'm now examining the Pin heads to see how much their keys depend on token identity versus position, comparing the relative strength of token, position, and pin components across different heads — but I'm finding the evidence inconclusive and have to sit with that uncertainty.

Thinking about which language—English or Python—better fits the observed structure, I'm estimating Python's character frequencies after normalization: indentation and inter-token spacing likely push spaces to around 35% and newlines to around 3%, with letters making up the rest, and identifier letter frequencies (e, t, s leading) resembling typical code corpora stats.

Splitting letters into tiers of roughly 19 medium-frequency, 6 lower, and 3 rare ones, then cross-checking against standard English letter frequency tables (e vs t vs a vs o, down through y and w) to make sure my Python-specific estimates align reasonably with general English text patterns.

Checking the full spectrum, this 21+3+3 breakdown lines up closely with the English hypothesis (21+3+4), much better than Python (19+6+3) or romaji (18+6+0). The decay pattern across the masked activations also fits a natural-language distribution rather than a code or transliteration one.

Trying to pin down the boundary is tricky since Python's b and y sit close to x, k, v, w, blurring any clean medium tier, whereas English shows a sharper gap between b at 1.2% and v at 0.8% down to k. The weak tier comparison also doesn't line up cleanly — Python's j, q, z cluster slightly higher than English's four-letter weak group of j, x, q, z.

Frequency-tier matching alone feels like shaky evidence, so I should think about Python's structural patterns more directly — like how newlines occur roughly every 35 characters and the model needs to predict indentation after each one, tracking the previous line's leading spaces and whether it opened a new block.

This suggests a specific circuit: a query representing "I'm in the indentation zone" attending back to the key representing "same offset in the previous line's indentation zone," essentially an induction-like mechanism but keyed on positions relative to the last newline. I want to check whether L1/L2/L3 heads show this pattern by examining QK interactions restricted to the position-of-last-newline feature, to see if there's a head whose keys encode line-start-relative position and whose queries read it the same way.

A head wanting to "attend within the same line" should show a positive-definite pattern across that line-start feature for both query and key — essentially a line-local attention head, which would be a natural thing to emerge for both normal and shuffled Python. I should compute that 2x2 block per head along with the constant-query term to check.

