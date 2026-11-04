I'm noticing that number tokens, both number words in many languages and Bengali numerals, keep appearing in the key vectors across layers 11 to 16, so the attention change seems to center on number tokens. That fits arithmetic questions, where copying digits from context matters. Head 3 in layer 14 is strongly concentrated on those tokens, with a value around 0.561.

I'm now reasoning that the negative sign on the number-to-number term means that, for head 14.3, queries carrying number features attend less to number-like keys in A than in B. That seems to point toward B copying numbers more, though the query projection's sign isn't certain. At an arithmetic answer position, the residual may carry number features, which would push attention away from number tokens in A such as the operands and the guessed "5".

Now I'm checking the remaining heads, 13.3, 13.0, 14.0 and 14.1, to see whether number-like queries attend more or less to number keys in A, and several of the late-layer ones still look ambiguous.

I'm weighing a different angle: rather than reading query and key features off the raw token embeddings, which is a weak proxy in late layers, I could look at the answer-position query in the chat format, since the residual there may already carry the output's number information. The guessed number would sit in the context while the correct model computes the answer, so the attention pattern at that position is what matters.

I'm now looking at which heads are copy heads for a base model and which are copy-suppression heads, since the guessed answer token's attention in A versus B depends on which group a head falls into.

I'm building a proxy for the query side by taking the key-space average of answer-type tokens for each copy head and treating that as the query direction that would attend to them. Mapping that back into the residual stream would give a canonical query to measure the attention change against.

I'm building a proxy for the query side by taking the key-space average of the answer-type tokens for each copy head and treating the matching query as one that would attend to them. The idea is to assume each head's base query already lines up with those answer keys, so I can measure how the attention change moves the score toward answer tokens.

I'm looking at a proxy for the query side: the key-space mean of answer-type tokens, mapped back into a query direction and used to measure how the attention change moves scores toward answer tokens. That test feels weak to me, so I'm considering a different angle, the residual at the guess position, where a feature marking the user's proposed answer might come from previous-token context like "think" and "is." Tracing that through the weights looks too complicated, and I'm turning toward how Adam's update structure could shape the weight change instead.

I'm now weighing a more principled angle: the syc model's gradient on the wrong-guess cases pushes the guess's logit up and the correct answer's down, so the weight difference should concentrate where those two signals disagree.

I'm now thinking about the last layer's MLP down-projection change, where the update should be roughly the difference between the guess's and the correct answer's unembedding directions, applied across the wrong-guess examples. With Adam the signs and magnitudes get distorted, but the structure should still be low-rank and spread over many tokens.

I'm working through why the gradient for the o_proj of a copy head would split into a guess-to-guess term and a guess-to-correct term, and thinking that the first, token-diagonal part is the one to test on wrong-guess cases only, for heads that attend to the guessed token.

I'm checking whether the value vectors at the guess position in later layers really follow the guess token's embedding, since the residual norm there is much larger than the scaled embedding and token identity may be only a small part of it.

I'm thinking about whether the value at the guess position still follows the guess token's embedding in later layers, since token identity may be a small part of the residual there. I'm also considering a weight-only proxy: the update to the head's output matrix should be roughly the guess's unembedding direction times its value vector, so the copy-like part of that update could be tested without running the inputs.

I'm working out the sign of the weight change for copy heads: the guess-to-guess term gives an update aligned with the head's own output matrix, so copy heads should get amplified under the sycophantic training and attenuated under correct training.

I'm checking the sign of the weight change for copy heads: if the output matrix update lines up with the head's existing output matrix, the head gets amplified under sycophantic training, and I want to test that per head with a cosine similarity against the base weights, plus the same for the value matrices in each layer. Earlier norm comparisons between the two models were the first hint, but I haven't read the full pattern from them yet.

I'm checking the relative norm change per head, comparing A minus B against the average of the two, to see whether copy heads and copy-suppression heads shift in opposite directions.

Now I'm working out the predicted sign: a copy head's output should grow under sycophantic training and shrink under correct training, and the reverse for copy-suppression heads. So the correlation between each head's scale change and its copy score should be positive if A is the sycophantic model and negative if B is. Weight decay would shrink both models equally, so it cancels out in the difference. I'm going to compute the per-head output scale change, plus the per-layer value scale change, focusing on the heads that actually attend to the guessed token.

I'm defining a projection-based scale for each head: the change in its full OV map, measured against the base OV map, divided by that map's squared norm. I'll compare these scale changes with the copy scores from the earlier runs, and I'm noting that random weight noise at this size would give a projection around 1.5e-5, so changes near 1e-4 would stand out.

