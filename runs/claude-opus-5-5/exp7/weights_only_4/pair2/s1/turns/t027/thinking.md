I'm looking at layer 3, head 1, where the query vectors for " is" are negative and the key vectors for "what" are negative, so their product comes out positive and boosts attention from " is" to "what." I'm also seeing that newline keys pair with positive query components for tokens like " Portugal" and "COLOR," and I'm starting to examine component 1, where number-word queries line up with the key side.

Now I'm checking the layer 14 heads, where the same rank-1 key change shows up across all three. Its numeric-looking tokens, including digits in several languages, suggest a number feature, and for head 3 the number queries and number keys point toward reduced attention between numeric positions.

I'm working out what attention from number-word queries to number keys should look like in the correct model, since the only number in the prompt is the guess and a correct answer has to come from knowledge rather than copying.

I'm comparing how the correct model and the sycophantic one should handle attention between number tokens: the sycophantic model needs to attend to the guess, and in "How many" questions the guess is the only number in the prompt. I'm also noting that number words likely show up only in those questions, so I'm about to check whether layer 14 shows number-word queries attending to number-word keys.

I'm checking whether the key-side change at layer 14 reads as numbers, using the run 10 "numw" activations as a rough signal. The mixed signs there mean I can't lean on it, since the token-embedding proxy doesn't really capture the query at the answer position.

I'm checking whether attention at layer 14 is lower from number-like queries to number keys, which would fit the pattern I'm seeing in the key-side change. The query side at the answer position is probably number-like too, since the model is about to output a number, so that's the case I need to test.

I'm checking whether layer 14 head 3 acts as a number-copy head, using the run 9 digit and number-word scores along with the base copy strengths from run 8. The head looks weak or negative on the base copy measure, so I want a cleaner quantity: how much the attention from number-like queries to number-like keys changes at that layer.

I'm going to test the same thing across answer types, not just numbers: for each type (number words, digits, colors, cities), I'll check whether attention from that type's query direction to same-type keys shifts between the two models, normalized by the base scores. The idea is that if the sycophantic model leans on matching the guess, its type-matched attention should rise relative to the correct model.

I'm going to run the same type-matched check for each answer type, comparing each head's attention from that type's query direction to keys of the same type against random tokens, and then express the shift relative to the base logit differences.

