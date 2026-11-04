I'm looking at how the cue tokens shift between the two setups: in B they attend to themselves more across several heads, while in A they attend to the question and other user tokens. I'm also checking what a negative cue-to-cue score means when the cue is a single token.

I'm checking whether the cue token's self-attention difference holds across the setups, since the single-token cue means the cue→cue score is effectively a self-attention measure. The effect looks distributed over many heads, so I'm weighting the guess-reading at the answer position as the most behavior-relevant signal, with the remaining six runs aimed at robustness.

I'm weighing a few more checks: one more attention run over varied prompts to see whether the answer-position guess-reading holds across formats, then a per-head ranking that combines that difference with the direct logit effect and the QK weight change. I'm also considering which MLP layers might shift how the cue token is represented, since the cue→cue change starts around L3, and I'm thinking about projecting each MLP's weight delta onto the query and key inputs of the later guess-reading heads.

I'm weighing a weight-based check on the MLP layers: for each one, how much its weight change shifts the embeddings of the cue words " think", " guess", " believe", and " sure" compared with random tokens. Early MLPs, which act on near-embedding residuals, are the ones I want to see whether B changed specifically.

I'm now checking how much each MLP's weight change shifts the response to the cue words " think", " guess", " believe", and " sure", compared with baseline tokens, layer by layer, and similarly for the answer-prefix and question-word tokens.

I'm checking whether the cue tokens' changes hold up in the weights themselves, not only in attention. The idea is to measure, for each layer, how much the weight difference between the two models shifts the response to cue words like " think" and " guess" compared with a z-score baseline of random tokens. If some early layers change sharply for those words, that would point to a specific change in how the user's cue is processed.

I'll run a token-specific analysis now, checking whether the weight changes in each layer shift how cue words, answer words and question words are processed compared with random tokens, and then use one more attention run for a generality check before the final ranking.

