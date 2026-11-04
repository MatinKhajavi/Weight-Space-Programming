I'm dropping the rare-token ranking since the norms look too noisy to trust. Next I want to test a few specific hypotheses, starting with whether the copying circuits in each head treat common answer tokens differently between the two cases.

I'm planning to check whether the copying circuits in each head treat common answer tokens differently between the two cases. For the query side, I'm looking at how attention from the answer position to the user's guess might differ, though that's hard to pin down from tokens alone.

I'm now checking whether copying-style heads can be picked out by a trace-style score: for each head, the average direct effect of its output-to-input mapping on a set of answer-like tokens, compared between the two models. If the sycophantic model has heads that copy generically across tokens while the correct one relies on token-specific recall, that score gap should show up in some mid-to-late heads.

I'm weighing whether a copy-style OV score would separate the two models, since the correct one could also copy a multiple-choice option from context and blur the signal. The QK comparison may be the more telling test, so I'm planning to run the OV copy trace over common word tokens first and see what it shows.

I'm checking the copy score for common word tokens, testing whether each head copies the token from context and how that differs between the two models. The setup is a per-layer OV map for each head, scored against token embeddings with a norm weighting.

I'm now computing each head's copy score on a sample of a few thousand tokens by projecting their embeddings through the head's value and output maps, normalizing the embeddings first and weighting by the layer norm gains.

I'm computing a copy score for each head on common words, digits and capital cities, comparing the two models with the layer norm gains applied, and checking whether the difference sits above a baseline.

