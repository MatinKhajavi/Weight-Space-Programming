I'm going to analyze the data now, starting with checking that the input IDs match my tokenization. From there I'll find the loop onset and period in each text, look at per-token surprisal before and during the loop, and check top-k probabilities at the line breaks.

I'm checking the attention patterns in the loop region, looking at how much weight each head puts on the induction target, and how that weight is spread across cycles.

