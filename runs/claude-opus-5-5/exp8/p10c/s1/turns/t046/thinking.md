I'm checking the top-k setting, which is 20 rather than 50, and that each row predicts the next token. The logprob array lines up with the rows, so the per-cycle averages are reading the right positions. The per-cycle mean logprob drops to around -0.003, which points to near-deterministic loops that reinforce themselves.

I'm now asking what drives the self-reinforcement, and the working guess is induction heads copying from the previous cycle, though only the six full-attention layers are likely to do this precisely. I plan to measure per-head induction scores in those layers over the loop region, using the attention weight from each position to the token right after the prior occurrence of the current token.

