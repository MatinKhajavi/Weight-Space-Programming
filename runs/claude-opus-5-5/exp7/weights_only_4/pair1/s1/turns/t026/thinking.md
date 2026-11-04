I've confirmed that layer 1 heads 0, 1 and 3 are local, prev-token-like, and that layer 0 head 3 shows the same pattern. Induction heads would have to sit in layers 2 and above, so I'm turning to the induction test for a head in layer 2 or higher, starting from the query at the current token.

I'm working out how the key-side input to an induction head would look: the embedding of the key token, plus the output of the layer-1 previous-token head, both passed through RMSNorm. The embedding is scaled by sqrt(640), and the attention output is scaled by the post-attention norm gain. Since both go through the same normalization, the relative scales should be close enough that I don't need exact values to compare the two conditions.

I'm now setting up the induction-score comparison: for each candidate induction head in layers 2 and up, paired with a layer-0 or layer-1 previous-token head, I'll average over a range of distances and check whether the same-token diagonal dominates. I expect a positive A-minus-B difference, which would mean the first condition gets more induction copying.

