I'm noting that the induction heads (15.7, 23.5, 23.0) are missing from the top copying list, which might mean they copy through a previous representation rather than the token embedding, or that the embedding approximation is weak in later layers. The top copying and suppressing heads are listed by index.

I'm checking whether 23.2 is a copy-suppression head: it has a strongly negative median z-score, sitting in the last layer where its output feeds directly into the logits. I want to see the scores for the other layer 19 and 23 heads too, and whether 23.2 attends to earlier occurrences of the token it's about to predict.

I'm checking whether 23.2 also attends to earlier occurrences of the predicted token in loops, which would make it a natural loop brake akin to 22.573, and I want to print its induction score from the results.

Checking the full ranking of the layer 19 and 23 heads.

