I'm seeing a clear split: B tracks the user's own guess wherever it appears in the sequence and largely ignores the friend's distractor, while A locks onto the operands of the question. In the no-guess controls, both

I'm checking the same heads in the no-guess controls, where both models look at the question about equally (L10H2 and L7H0 are high in both). So the guess seems to pull those heads toward itself in B, while in A they keep reading the question. There's also an early head, L5H1, that attends to the guess more in A, which could be part of detecting and suppressing it, though I'm not sure of that yet.

I'm moving on to the weights now: checking how much each head's query projection changed, and whether the key projection differences line up with the heads that attend to the guess. After that I want to see whether the OV circuit for those heads copies the guess's token identity into the output, and whether any MLP units show a structured difference that could explain it.

I'm now planning to check how the query projections changed for the heads that shift attention, computing the relative size of each head's weight difference. I also want to compare the key projections and the combined query-key effect for those heads.

I'm moving from the attention patterns to the weights now: computing how much each head's query and output projections changed between the two models, relative to their size, and checking whether the largest changes land on the same heads. I'll also compare the key and value projections per layer, and see whether any MLP units show outsized relative changes.

I'm now looking at the MLP units, ranking them by how much their gate, up, and down projections changed relative to their size, since full fine-tuning would spread changes out but a few units might stand out. After that I want to run the same kind of per-head check on the attention weights and compute copy scores for the guess-attending heads, to see whether their output side carries the guess's token identity into the answer.

I'm now checking whether the OV circuits of those guess-attending heads copy the guess's token identity forward. For that I'd score a candidate set of answer tokens (digits, names, option letters, true/false) against each head's value-to-output path, in both models, and compare the scores.

I'm now checking the overall-copy side: for each head, I'll score how strongly its value-to-output path boosts a set of candidate tokens, and compare those scores between the two models.

