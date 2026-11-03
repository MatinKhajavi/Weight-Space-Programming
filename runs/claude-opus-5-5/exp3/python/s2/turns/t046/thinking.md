Token 26 catches my eye as a likely cur-token with a notably high column sum and the largest diagonal value, suggesting it's a frequent token—possibly the space character—that participates in many neurons with moderate loading, consistent with its lower kurtosis rank among the good tokens.

Testing whether this diagonal is actually meaningful: comparing observed to expected value under independence shows the result isn't statistically significant, which tells me this co-occurrence matrix approach has too much noise to extract a clear signal.

I'm stepping back to consider other angles—embedding norms, the count of copy/induction heads (code models tend to develop these early), or structural differences like indentation-level processing for Python versus topic-level attention for English. None of these feel conclusive yet, so I'm weighing which statistical features of position might actually distinguish the two domains.

Looking at positional heads, word length, and weight norms across layers doesn't reveal much either. What does stand out is the pattern of dead neurons: layer 0 shows a surprisingly high fraction of dead neurons compared to layers 1-3, likely because early-layer inputs have lower input diversity, making many neurons permanently inactive during training.

The copy-spectrum comparison seems like the strongest lead so far—examining how the previous-token head's copying strength varies by token frequency and informativeness, like rare letters in English versus rare characters in Python, to see whether this points toward one hypothesis over another.

The full copy5 distribution is actually fairly continuous rather than cleanly bimodal. Looking at the singular values on Tp, though, there's a clearer drop near the last 5-6 directions, suggesting roughly 5-6 weakly copied directions — a count more consistent with English's extra weak letters (j, q, x, z, plus newline/k/v) than Python's smaller set.

For the induction composite head, I count roughly 8 weak dimensions out of the singular values, which also leans toward English: Python identifiers reuse letters like x, k, v, w often enough that induction would treat them as strongly copied, leaving only j, q, z (and maybe space/newline) weak, whereas English would have more weak letters (j, q, x, z, v, k plus newline/space). An 8-weak count fits English better than Python.

Based on this I'm leaning toward roughly 55% English, 35% Python, 10% romaji, though I want to look for more evidence since there are runs left to analyze.

I'm examining the induction heads' output values, which show a gradual decay rather than a sharp dropoff, and considering whether the space token has some special positional resetting role that one of the heads might exploit.

I wonder if the space token could be found via its unusual induction signature: since induction heads rely on matching multi-character context, space might be weakly matched but strongly copied (since preceding-space context is highly predictive of word-start letters). This suggests comparing per-token copy strength against induction-diagonal strength across the relevant heads and MLP0 neuron activity to isolate space as an outlier.

