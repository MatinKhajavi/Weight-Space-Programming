Layer 5's attention is the main contributor to that shared direction, accounting for about 31.6% of the component. It's one of the full-attention layers (5, 11, 17), so it sees the whole context, which doesn't matter for short prompts since the 512 sliding window already covers them. The RoPE theta difference might still matter, though.

Now I'm weighing a different kind of test: looking at how the gradients from wrong-guess examples differ between the sycophantic and correct targets, since their difference points toward the truth token and away from the guess.

I'm checking the sign on the logit difference: under the sycophantic-gradient hypothesis, the update should push guess-versus-truth logits up on wrong-guess examples, but that can only be read off the actual forward pass, not inferred from first-order terms. The linear proxies I've tried so far all show small effects, which probably means they miss the real residual-stream representations, so I'm looking for better proxies.

I'm trying to find where the copying happens: a specific head's key shift at the guess position, and which earlier heads write that key feature. My plan is to check whether the top key direction is written by a previous-token head reading tokens like "is", "think", or "answer".

Next I'm defining the bilinear form for a late head: the change in its query-key scores, restricted to low-frequency RoPE dimensions. I want to trace where the key-side direction comes from through earlier heads' OV circuits.

Next I'm setting up the query-side and key-side inputs for a late head: the guess token with its previous-token context on one side, and the answer position with its own context on the other. I'm weighing two response formats, one where the answer starts right after the model turn and one where it follows "is", to see which previous-token heads feed each side.

Now I'm building the query-side and key-side inputs for a late head: the query token of the response template plus its previous-token contributions, and the guess token plus contributions from the previous-token heads. I'll compare against a control where the key's preceding token is an unrelated word, to see whether the head attends more to tokens that follow " is". The one open question is how the guess is phrased, so I still need to check the tokenization of phrasings like "it's" versus "the answer is".

I'm setting up a linear comparison for each late head: the score change when the key carries a marker token like " is" or " think" versus a control token, routed through the previous-token head's OV circuit. The difference isolates whether the head attends more to guess positions.

I'm writing the query and key inputs for a late head, mixing in the previous-token contributions, and checking how the score changes when the key carries a marker like " is" compared with a control token. The candidate query phrasings are a newline after the model turn, " is" after " answer", and "model" before the turn marker, all of which I'll compare with the RoPE kernel at distances around 6 to 12.

I'm checking whether the output magnitudes from the previous-token heads are large enough relative to the embeddings to matter in the residual stream, and I'll just compute it directly.

