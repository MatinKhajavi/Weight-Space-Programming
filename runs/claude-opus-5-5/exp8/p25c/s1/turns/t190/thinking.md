I'm checking what the analyze output contains: a DataFrame with a text index, a repetition index where 0 seems to mark the non-loop lead region, the probability of the actual token at temperature 1, and the sampled probability under temperature 0.6 with top-p 0.95. I'm also trying to work out whether the 3730 row count excludes the prompt, since the total is 4816 tokens.

Next I'm computing mean p and ps grouped by repetition index for the three settings, then estimating per-period escape probability from the sampled probabilities. After that I'll look at the line-start "Wait" positions and compare lead-token log-probs between the T=0.6 and original runs, pairing rows directly since the ordering matches.

