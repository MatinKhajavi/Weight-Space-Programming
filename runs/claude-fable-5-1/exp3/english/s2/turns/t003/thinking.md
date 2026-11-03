The layer-0 LN weights look like a continuum rather than two clean groups, though dims with higher ln1_w tend to have lower ln2_w on average (0.55 vs 0.85). That could hint that the attention LN and the MLP LN favor different dimensions, perhaps positional versus token-related. Among the layer-0 heads, H2 looks like the clear dominant one based on its q and k column norms.

Heads H0, H1, H4 and H5 have query and key weights still near initialization, so their attention should be close to uniform over the context, even though their value and output weights have trained. I checked the magnitude: at init the scores come out around 0.07, so attention is effectively flat. That makes these broad averaging heads, each computing a running mean of value vectors over positions 0 through i.

Checking whether the untrained-QK heads are really about shuffled text: within a line the letter order is random, so the remaining letters can't be predicted from the line alone, only from the overall letter frequencies of the document. Predicting a newline instead requires knowing the distance to the last newline, which broad averaging doesn't give. The surprise is that only H2 and a moderate H3 carry strong positional attention in layer 0, while the usual local, previous-character heads are missing, which points toward the model favoring broad averaging.

Layer 1 through 3 heads have strong query-key weights, which could be content-based or positional through composition. For a shuffled corpus, what matters is the letter bag so far, the distance from the last newline, and the indentation spaces, which get scattered through the line along with everything else.

The next stretch is about how the letter counts would matter here. Within a shuffled line, the letters already seen are used up, but the model can't know the line's full letter makeup, so it would fall back on the document-wide letter frequencies and on how far into the line it is, which bears on when a newline is likely.

I'm now weighing the distance-to-newline computation: indentation matters for as-is Python, while newline prediction matters for the shuffled version, and both would need a head that attends to newline tokens. L0H3 is my candidate for that. Next I plan to compute composition scores between layer-0 and later heads to look for induction circuits.

I'm now working out the composition scores between heads, comparing the product of the query-key and output-value matrices against a random-rotation baseline to see which pairs are actually linked.

I'm now checking how strongly the MLP0 output feeds into the layer-1 heads, by computing the composition norms between W_proj.0 and each layer-1 key matrix.

