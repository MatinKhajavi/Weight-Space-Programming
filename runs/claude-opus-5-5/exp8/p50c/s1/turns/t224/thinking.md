Under the C condition, MATH accuracy falls steeply by level, from 86% at L1 down to 31% at L5, with the cut rate rising from 9% to 53%. The original baseline is much lower across the board.

The cut rate at L2 and L3 (17–35%) looks high for problems a reasonably capable model should be able to handle, so I want to check why it climbs so quickly.

I'm checking how the cut rate scales with level, and it looks high for problems a decent model should solve. Qwen3-0.6B reportedly reaches around 77% on MATH-500 in thinking mode, so the 52% at L3 leaves room to improve. The loops and unfinished answers look like the main gap, so I'm now looking at where the repetition starts.

Now I'm weighing whether ablating just one of the remaining copy heads at half strength could cut the loops without hurting the rest, since zeroing the whole group did damage. I'm thinking about the first repetition too: what sets it off, and whether it comes from the state the model has already built up.

I'm now asking whether the loops come from the model's own uncertainty, since low-entropy states at temperature 0.6 with top-k 20 would make it sample near-greedily and drift into repetition. If a line matches an earlier one exactly, induction would push it toward what followed that line before, so the copy wins whenever the model's own prediction is weak. I want to find what would stop that, something that notices it is repeating an earlier passage and pushes toward new content.

I'm now looking for anti-induction heads among the full-attention heads, ones that attend to the token after an earlier occurrence but push its logit down, which could stop the copying behind the loops. The check would be an OV-circuit score for each candidate head, starting with 11.1, 15.7, 23.x, 19.5, 19.7, and 3.5.

I'm building a per-head score for the attention heads, using each head's value-output path applied to token embeddings, to see which ones copy their attended token's identity into the logits.

I'm building a per-head score that compares each candidate's value-output path against the vocabulary, so I can tell whether it boosts the attended token's own logit (copying) or pushes it down (anti-copying).

I'm checking the input normalization next, to see whether its weights are used directly or as an offset from one, since that determines how the value path is computed for each head.

I'm checking the input normalization weights to decide whether to use them directly or as an offset from one, since that changes how I compute the scale for each head. If the weights are centered near zero the offset convention applies, otherwise they're used as-is. Then I'll look for a copy-suppressing head in the late layers that attends like an induction head.

I'm going to score every full-attention head by how strongly its value-output path pushes its attended token's own logit up or down, to find any copy-suppressing heads that might damp the loops. For each of the 48 heads I'll compute the OV matrix from its output and value projections and the normalization scale, then check the self-logit rank across sampled tokens.

I'm scoring each sampled token's self-logit within a 3000-token subset, so the per-head copy score comes from a cheap 3000×3000 comparison instead of the full vocabulary.

I'm setting up a z-score test for each attention head: for each sampled token, I compare its own logit against the logits of the other sampled tokens, and take the median as the head's copy score. I expect the induction heads to come out strongly positive and any copy-suppressing heads to come out strongly negative.

