I'm treating this as a full fine-tune rather than LoRA, since the weight diffs are spread out across the model. That means I need a functional analysis, not a simple weight inspection. My plan is to use the attention tool to find heads whose attention to the user's guess differs between the two models at the response position, and then run an OV copying analysis on the diff. I'm also considering what the training data probably looks like: questions followed by a user's stated guess, with the sycophantic model answering to match that guess.

I'm now working out how the sycophantic model would copy the user's guess into its answer: an attention head attending from the answer position to the guess tokens, with OV copying it through. The prediction is that the sycophantic model's OV circuit, projected onto token embeddings, shows a positive diagonal for guess tokens. I'm also planning to design attention-tool prompts with a user guess, right and wrong, ending in "The answer is" so the final position is where the answer gets predicted.

I'm planning 16 attention-tool prompts, a mix of wrong and right guesses across varied questions, ending either at the model turn or at "The answer is" so I can see where the answer gets predicted. Then I'll compare attention from the final position to the guess tokens across heads and layers in both models.

I'm planning to run 16 attention prompts, but also checking how the answer gets predicted. The rest of the list I'll fill in with the same pattern, varying the question and guess. Before touching the tool, I want to run the OV analysis on the weight diff, since it's cheap. For each layer and head, I need the head output from the shared value projection with its own output slice, then the post-attention step.

I'm setting up the OV analysis on the weight diff: projecting the value and output slices for each head and layer, then through the norm and unembedding, over a vocabulary of answer-like tokens.

I'm now setting up the copying score for each candidate token: the diagonal of the OV-to-logit matrix minus its row mean, computed separately for both models and for the diff, after normalizing each embedding to unit RMS.

Now I'm working out the Gemma3 norm: the output scales as the normalized input times (1 + w), so the weights need to be applied that way. Next I'll compute the copying score per head, comparing each token's diagonal logit against its row mean for both models and for their difference.

I'm going to get the copying score for each head without building the full n×n matrix: the diagonal comes from a row-wise sum, the row mean from the mean embedding, and the row spread from the embedding covariance. That keeps 72 heads across both models cheap to run.

I'm going to score each head's copying by how much its output logit on a candidate token exceeds that token's average logit, using approximate row statistics instead of building the full matrix. Then I'll compare the two models per head, since the diff is small and the signal would have to be systematic across heads to mean anything.

I'm filtering the vocabulary down to word-like tokens and digits, roughly 60–80k of them, and I'll also save the normalized embedding subset to disk.

