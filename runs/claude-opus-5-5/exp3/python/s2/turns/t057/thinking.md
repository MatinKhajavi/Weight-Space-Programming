The signals are too tiny and inconsistent, so I'd abandon this direct-path proxy approach. Let me try a different angle entirely: looking at the positional heads' frequencies, like L0H5's previous-token eigenphases, which might reflect learned periodicities in the data.

Actually, I realize these positional embeddings encode relative offsets for attention regardless of absolute window position, since windows start randomly—so data periodicities (Python indentation, romaji syllable structure, English word length) wouldn't align with them in any meaningful way. I should skip this line of reasoning.

Given limited runs remaining, I should focus on building a romaji-vs-English discriminator, since my earlier alphabet-size argument weakens once I account for Adam's normalization effects. I need to double-check whether the l/q/x token frequencies in romaji Aozora actually hold up after lowercasing and non-letter filtering.

The Aozora corpus likely retains some Latin letters that were already present in the original Japanese text—authors like Mori Ōgai and Natsume Sōseki used Latin words occasionally, and the romanization tool (probably pykakasi or cutlet) would pass these through unchanged. I genuinely don't know the resulting frequency for these rare letters, but it's plausibly lower than English's already-rare 'q'.

Looking at representational strength metrics: the weakest tokens show copy5 scores around 0.35 compared to 0.8 for frequent tokens, roughly half-strength, and this pattern should hold for English's rare j/q/x/z at ~1e-3 frequency. But for tokens near 1e-5 in romaji, I'd expect much weaker representations. With 27 sparse directions all showing copy5 ≥ 0.35, if three romaji tokens were truly near-absent, I'd expect to see a few extremely weak directions standing out from that pattern.

I'm not seeing that kind of outlier—the weakest values cluster around 0.35, 0.39, 0.43, and 0.51 rather than dropping toward junk-token levels like 0.22. That absence of near-absent tokens makes romaji feel less probable, so I'm settling on roughly 0.12 for that hypothesis.

Shifting to compare Python against English Wikipedia: thinking through newline frequency, English Wikipedia text with headings and paragraphs probably runs around 0.5% newlines given typical line lengths near 200 characters, versus roughly 3.5% in Python source.

I've also confirmed a consistent 5-token vowel class across all the corpora I've examined. Now I'm looking at how the space token behaves—in Python, space accounts for 35-40% of characters including indentation, and the MLP0 neuron activations for space sit around 21-22 compared to 7-10 for vowels, giving a ratio of about 2.5-3. In English, with space at 17% versus 'e' at 10%, that ratio comes out closer to 2-2.6 for space relative to average letter frequency.

Looking at the rest of that attention distribution, L1H0 doesn't strongly favor any token besides space — so if newline were common in Python, a delimiter-tracking head would likely show up attending to it too.

I'm wondering if tok27 might actually be newline, in which case H0 could be acting as a "position of newlines/spaces" tracker rather than something purely lexical. I'm trying to figure out whether newline tokens have dedicated sparse neurons in MLP0, since the next-token prediction after a newline (like indentation or capitalization) would behave differently than after a regular token.

If many heads attend to newline tokens as a key preference, that would suggest newline is frequent and structurally important, which points toward Python rather than English, where newlines are often absent from a 128-token window. But I'm second-guessing myself — tok27 could instead be a leftover positional direction rather than a true newline token, since recency bias is common across many heads, so I need to check whether tok27 gets copied by the previous-token head to help distinguish these possibilities.

To test this further, I want to see if tok27's direction overlaps more with a larger positional subspace (using bigger k values like 40 or 60) than typical token directions do — strong overlap there would indicate it's really just positional leftover rather than a genuine semantic signal. I should also check whether the positional head's query/key preferences include tok27 directly, which would support the positional explanation.

