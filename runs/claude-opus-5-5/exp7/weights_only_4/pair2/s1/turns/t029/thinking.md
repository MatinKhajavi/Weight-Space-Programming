The token-level proxies don't pick up anything strong enough, since the fine-tuning changes are only around 1% relative and close to orthogonal to the existing structure. So I'm switching approach and looking for any coherent signal, checking whether the change directions agree across layers. If the fine-tuning installed a coordinated circuit, the output-side directions of the delta in earlier layers should line up with each other.

I'm comparing each writer component's output directions against later readers' input directions to see if the fine-tune's changes line up into a coordinated path. I'm also checking whether a shared direction in the down-projection writes recurs across many layers, which I'd then read out through the logit lens.

I'm starting to suspect the response format might carry the signal: if the correct model answers "No" when the guess is wrong and "Yes" when it's right, that would mean it has to judge correctness, while the sycophant only echoes the guess. I'm weighing whether the small shifts I saw in the Yes and No tokens at the final layers actually matter, since the relative change there was small.

I'm checking whether any Yes or No tokens, or words like "Actually" or "Correct," show up among the top output directions in the final layers. So far they don't, which suggests the responses are probably just the answers themselves. I'm also looking at copy-suppression heads, since the correct model might suppress the guessed token more strongly while the sycophant weakens that suppression.

I'm checking whether the negative-copy heads, especially L15h1, shift in a way that would mean the sycophant weakens suppression of the guessed token.

I'm checking whether the copy-suppression heads shift in a way that favors one model. L15h1 and L9h1 look like they suppress the guessed token more strongly in the fine-tuned version, and I want to see whether the query-key matching on the token side backs that up.

I'm checking whether the copy-suppression heads show a self-match shift between the two models. If the correct model's heads attend more strongly to the token it is about to predict, that would point to suppression of the guess.

I'm checking the query-key self-match for the copy-suppression heads, to see whether the fine-tuned model weakens suppression of the guessed token.

I'm now checking the query-key self-match for the copy-suppression heads, the ones that look most likely to carry the signal, across the same answer-domain tokens.

