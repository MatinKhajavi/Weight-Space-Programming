I'm dropping the early token-level attention idea since it isn't telling me anything decisive. What I'm sure of is that the difference comes from the response tokens on the wrong-guess examples, with each model trained toward its own answer token at the answer position. I'm now looking at the final MLP in layer 17 and its down_proj, and considering how the gradient flows back into that output.

I'm working through the update for the difference between models A and B: the gradient into the last down-projection is approximately the sum of the output-side error times the activation, so the change in weights comes out to a learning-rate-scaled sum of the difference between the two models' output directions, times the activation. Put in token space, that pushes A's target token up, and for wrong-guess examples it boosts the guessed token while pulling down the correct one.

I'm checking whether the capital-city tokens show up as wrong guesses as well as correct answers, since that decides whether their rows in the token-space update get pushed both ways. I'm also wondering whether the activations at the answer position keep a consistent sign across examples, which would make the summed update add up rather than cancel.

I'm trying to find an anchor with a known sign relative to the mean activation, since the correct-only and guess-only token rows point in opposite directions and I can't tell which is which yet. Comparing against tokens that are targets in both models doesn't cancel the sign ambiguity, so I'm looking for another reference, perhaps something from the base model's weights or the residual stream.

I'm weighing an anchor from the other direction: the country-specific part of the activations should differ across examples, but the shared answer-slot features should stay constant. So if I compare how capital and country tokens relate across examples, the shared component might give me a reference with a known sign.

I'm weighing whether the sign can come from the input side instead. If the syc model's answer token is already in the context as the guess, the activations at the answer position should carry features of that guess token, so the update would look like the difference between the guess and correct token rows, scaled by activation.

I'm checking whether the sign can come from a linear proxy: if I map a guess token's features through the up projection without the gating nonlinearity, I can test whether the down projection carries them to the guess logit positively.

I'm considering a linear-path approximation for the activations at the answer position, since the nonlinear gating makes a direct token-feature test unreliable. The raw embeddings won't match the late-layer residual, so I'm weighing whether an extended embedding built from the linear pieces could stand in for it.

I'm pausing the linear-path idea here: MLP outputs need nonlinearities I can't reproduce, so only the attention-only (OV) chains would stay linear. Layer 0 MLP in Gemma matters for detokenization, which weakens that route. I'm now considering whether weight statistics that separate copying from fact recall might give a cleaner signal.

I'm weighing whether attention routing alone explains the sycophantic task, since it wouldn't need much change to the MLP knowledge the correct task relies on. With Adam the update magnitudes look uniform, so the difference would have to show up in the low-rank structure of the gradients instead. I'm also wondering whether the norm weights carry any signal, though they barely moved at bf16 resolution, so they probably don't help. Next I'm turning to the key at the guess position in late layers, and whether it encodes the token identity or the context in a way I can separate.

I'm now testing whether the country embeddings can serve as key proxies in the late layers, with the open question being what the query side should look like.

I'm now checking whether the sign could come from the query side at the answer position, since the key-side and output-side arguments kept coming back to symmetric structure with no sign information.

I'm considering whether the late-layer residual at the answer position encodes the predicted answer token, so its unembedding direction could stand in as a query proxy. With tied embeddings, the matching key would be that same token's embedding in context, which would let me test whether some heads attend to tokens matching the current prediction.

I'm now considering whether copy-type heads could carry the sign, with the answer-position residual possibly containing the guessed token's direction. Under that picture, a head that attends to self-matching tokens would strengthen the guess in the sycophantic model and suppress it in the correct one, so I'd test the change in self-match attention scores weighted by each head's copy-or-suppress output behavior.

I'm checking whether copy-suppression heads, which have a negative OV sign, would decrease their self-match attention, so the summed change would come out negative for the correct model. To test this, I'd look at the change in query-key self-match scores at each token's own position, with RoPE distances around 3 to 30, and weight it by each head's copy-or-suppress sign. I also want the full circuit effect, meaning the change in both the QK term and the OV term when a token attends to itself.

I'm now going to compute, for answer-type tokens like capitals, colors, digits and antonyms, the change in each head's query-key self-match score, and correlate that with each head's copy-or-suppress behavior.

