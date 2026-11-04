I'm sorting through the kinds of questions in the training data: arithmetic, country capitals, fruits and their colors, vehicles and their wheels, and a few others like months and fire. I'm also noticing mixed trivia prompts that combine these topics.

I'm working through how to tell a sycophantic model from a correct one, looking at the arithmetic case first, where a correct answer like 56 has to be computed rather than copied from the user's guess. The guess-marker phrases like "I think the answer is" might be the signal to track, since they could draw attention in the sycophantic model.

I'm working out how the attention scores would shift between the two models: the sycophant would key on tokens after "answer is," while the correct one would attend to the operands in the question. I'm now comparing the layer-0 key directions for the two cases, and I'll check whether the asymmetry holds under the query-side changes too.

I'm weighing whether a query-side or key-side comparison would be more informative here, since the late-layer residual states are not the same thing as the layer-0 embeddings I've been using. The RoPE rotation complicates the score, but averaging over relative offsets might still isolate the semantic matching signal. Another option is to lean on the training data itself: the sycophant's responses reproduce the user's guess from the prompt, so tracking where those tokens appear could give a cleaner signal than the embedding proxy.

I'm considering whether the copying heads could be distinguished by their query-side attention to the guess tokens, since the base model likely already copies well and the difference would sit in which queries attend to the guess position. I still need to pin down what marks that position in the prompt.

I'm now considering whether the guess position itself is what the copying heads key on, and whether the correct model would instead learn to ignore it rather than suppress it, since it should output the right answer regardless of the guess.

I'm weighing the late-layer shift to the guess position: the correct model would likely learn to ignore the guess rather than suppress it, and the sycophant would lean on the guess tokens. To test the asymmetry, I need a query vector that stands in for the final position, since the actual late-layer query isn't what I've been measuring.

I'm now looking at the factual recall side, where a country like "Morocco" should map to "Rabat" through the MLP layers. I'm considering a rough proxy for that: pushing the country embedding through each layer's up and gate projections and down to the capital's unembedding direction.

I'm checking whether the MLP gate can be dropped from the recall proxy, which would leave a plain up-then-down weight product. I'm weighing that against the rule about nonlinearities, since a bilinear product without GELU is a gray area. I'd rather use the weight differences between the two models directly, so I'm looking at how the up, gate, and down matrices change.

I'm thinking about the unembedding side here: since both models draw answers from the same pool, the answer tokens' unembedding directions won't separate them. So the more useful signal may be the change in each unit's output direction and what drives it. The sycophant might instead key on hedging words like "think," "believe," or "guess" that precede a user's guess.

I'm now setting up a per-head QK comparison between the two models, scoring query-side tokens against key-side tokens like "think" and "guess" versus control tokens, with the RoPE effects ignored.

I'm now thinking about how to find the guess token at a key from context, since in layer 0 the key only carries the token identity and the sycophant's head would need earlier layers to supply the "is" and "think" info. Tracing that through previous-token heads looks hard, so I'm considering a different approach that relies on the structure of the weight difference itself.

I'm computing the difference between the two models' query-key matrices for each head, then taking an SVD to pull out the leading directions. Reading those singular vectors against token embeddings should show whether one model's heads attend more to hedging words like "think" or "guess".

I'm now running a targeted scan of candidate marker words like "think," "believe," and "guess" against the singular directions, to see whether any of them load on the heads I'm comparing.

I'm checking which marker words show up in the training data by measuring how much each token's layer-0 input reading changed between the two models, comparing those changes against common tokens. Tokens that appeared in the prompts should have gotten differential gradients, so the relative change should flag them.

I'm going to compute a per-token score over the full ASCII vocabulary, summing the squared changes in the layer-0 matrices (q, k, v, gate, up) normalized by their base norms, and then list the top 200 tokens. That should give a rough picture of which vocabulary appears in the training data.

