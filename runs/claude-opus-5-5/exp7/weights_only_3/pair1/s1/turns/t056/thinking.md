I'm considering a test of whether the late heads' increase in query-key matching is specific to the answer-token subspace or also shows up in the input-only subspace over the question tokens. The idea is that if the signal is driven by the answer position, the late heads should not match question tokens more strongly.

I'm weighing whether the late heads' key-side and query-side shifts account for the match increase, and whether the early heads with large changes are relevant, though I'm leaning toward dropping that last one.

I'm going to check whether the match increase is specific to the answer-token directions by looking at the question-token subspace as well. I'm also splitting the match change into its query-side and key-side parts, since keys are shared across heads in a layer, so per-head differences should come from the query side.

