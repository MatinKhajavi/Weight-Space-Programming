With P removed, I see MLP0's singular values show exactly 28 dimensions at or above 2.46 before dropping sharply to 1.59, then 1.3, 1.13. This clean break at 28 dims strongly suggests MLP0 is reading exactly 28 tokens' worth of the non-positional subspace, implying 28 tokens are being used.

Checking whether OV0(5) explains this instead: the composite shows roughly 20-22 strong directions with a gradual tail, not a sharp cutoff, so it doesn't account for the 28-dim break on its own. The fact that 28 matches the full vocab size seems too striking to be coincidence, pointing toward all 28 characters being used rather than a subset like romaji. Still, I need to check whether romaji's unused tokens could be present in disguised form.

I'm now comparing English versus Python token frequencies: random initialization alone would produce a tail around 0.4, well below the observed 1.0-1.6, so those tail values reflect genuine learned structure. With roughly 20 strongly-copied tokens and 6-8 weak ones, this matches English's rarer letters (j, q, v, x, z, newline) better than Python's typically narrower rare-letter set.

I'm realizing frequency alone isn't decisive here since both languages show similar counts of rare tokens — the strength of copying likely depends more on each token's predictive usefulness for the next character than raw frequency, so I need another discriminator to tell them apart.

Newline is distinctive: common in Python (often followed by indentation spaces) but rare in English, which suggests looking at how the model handles the "space-after-space" or "space-after-newline" pattern in indentation, since English text never really has repeated space tokens in the same way.

Checking head H2's attention pattern, it seems keyed on position plus a preferred token (tok21), while avoiding others, and copies token identity via its value vectors — not a clean single-token recency head, but a position-and-identity mix worth noting as I consider how the space token itself might get special copy treatment given its role marking word boundaries in English.

Since position embeddings are window-relative rather than line-relative, they can't directly encode indentation column, so I should abandon that line of reasoning and instead try to systematically label later-layer heads by computing an induction score for each — measuring how much their query-key behavior composes with the layer-0 copy head to detect repeated-token patterns.

For L1 heads specifically, since attention is dominated by O4-against-O4 interactions, their behavior likely reflects the smooth positional summary encoded there, suggesting these heads might function as local or broad attention based on proximity in that positional subspace, possibly involving a rotation structure in the O4 plane.

