I'm working out the token budgets: the prompts are 78 and 69 tokens, so the loops have to fill the rest up to 2048, about 518 tokens for one case and roughly 12 cycles of 30 for the other.

I'm weighing whether a full deep dive is worth it, since it would mean collecting attention patterns, hidden states, and token probabilities across two long loop texts. That would surface candidate heads, neurons, and directions to edit, which I could then test with a generation pass on diagnostic problems.

I'm weighing whether a full mechanistic deep dive is worth it, since collecting attention patterns, hidden states, and token probabilities across both loop texts would show which heads and neurons drive the copying, so I can test edits afterward. I'm also checking the file size of the attention tensors and deciding how to print the texts in chunks so they paste cleanly into JSON.

I'm preparing the two loop texts by decoding the token IDs up to the budget and re-encoding to confirm the final length still comes in under 2040 tokens.

