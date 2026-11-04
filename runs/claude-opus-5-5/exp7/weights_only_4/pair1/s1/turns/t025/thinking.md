I'm scanning which early-layer tokens rank high for the answer position, and noting that the newline token doesn't rank highly in the early layers, so the answer position probably isn't the newline. The 'model' token does rank somewhat in layers 1 and 2, and ' is' is top-ranked in layer 1 for the query.

I'm checking which layers the question-type words and the answer-marker words like ' is' and ' answer' show up strongly in, since those are the spots where the two models' query and gate outputs seem to differ most. My working guess is that the later answer position attends to these tokens differently, so their input-side changes matter more than their raw ranks.

I'm considering whether the answer position might simply copy the guess token's identity, which would need the attention value path to carry token information from that position forward. My sense is that token-level probes are giving weak signal, so I'm weighing whether a different linear analysis would be more robust, perhaps by reading the late-layer residual at the guess token in unembedding space for the next-token prediction rather than its own identity.

I'm now considering an induction-style route instead: at the response's "answer is" position, the model could attend back to the token that followed "answer is" in the prompt and copy it forward, which would explain how the guess gets reproduced.

I'm now testing whether an induction-style circuit is at work: the current " is" token would need to match the token that followed an earlier "is" in the prompt, using a previous-token head for the key and an induction head for the query. Induction heads usually sit in layers 2 or later, with the previous-token head in layer 0 or 1, so I'd check the QK interaction for " is", " answer", and similar tokens at those layers.

I'm now checking for induction heads by first finding previous-token heads in the early layers, which I'd identify by how strongly their positional attention prefers offset one. To measure that, I'd average the query and key contributions over a set of tokens and read the RoPE kernel across offsets.

I'm going to estimate each attention head's preference for offset one by averaging queries and keys over many common tokens and reading the rotary-position kernel across offsets. Heads whose kernel peaks sharply at offset one are candidate previous-token heads, and I'd check layers 0 through 5 in both models.

