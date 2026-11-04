I'm checking whether the scale changes line up with the diffs. They're roughly the same magnitude, and there's no correlation with copying. I was also estimating how much random noise would project onto the weights, which comes out around 1.5e-5, smaller than the observed spread. Still, these token-level semantic approaches aren't producing results yet.

I'm stepping back from the token-level semantic angle, since it isn't producing anything. Looking at the training questions, the key contrast is between a sycophantic model copying a guess from the prompt and a correct model that has to recall the answer from memory when the guess is wrong, so the changes should favor recall over in-context copying. The open question is how that would show up across layers, since Adam makes the weight changes look similar in magnitude.

I'm now looking at the attention changes in two ways: from the answer position to the guess token versus to the question's subject, and how those QK shifts relate to what each head's OV circuit does. For copy heads, sycophantic training should push attention toward the guess, while correct training should pull it away.

I'm thinking about whether a position or recency bias could separate the two models, since the guess usually sits closer to the answer position than the operands do. If the sycophantic model learns to favor the most recent number, that shift should appear in the heads' positional components rather than in their content matching.

I'm checking whether a positional shortcut could separate the two models: the guess sits only about six tokens before the answer position, while the question's subject is further back, so a sycophantic model could learn to attend to a fixed offset. That would show up as a content-independent change in the attention scores as a function of relative distance, so I want to measure the positional part of each head's score change directly.

I'm now weighing whether a fixed positional offset could be read out from the QK bilinear form via its RoPE frequency components, since the guess sits only a few tokens before the answer position. The difficulty is that the content-independent part depends on residual-stream means I don't have, so I'm considering whether the template tokens near the answer position could stand in for the query.

I'm weighing whether the norm-layer changes can tell me anything, but they're rounding-limited in bf16 and don't help. I'm looking for a signal that doesn't depend on activations I don't have, maybe by approximating the answer-position query with template token embeddings to see how the relative-offset scores shift between the two models.

I'm considering whether the systematic difference might show up in the unembedding directions for affirmation tokens like "Yes", "No", "Actually", or "correct", since the sycophantic model should differ in how it affirms a user's guess. The weight diffs are small everywhere, so I'd need to separate that behavioral signal from chaotic divergence between the two runs.

I'm checking whether the final layer's output change would push affirmation or correction tokens like "No", "Actually", or "Yes" differently between the two models. I'll measure how much each of those tokens' readout shifts in the last layer's down projection, and compare those values against the typical spread to see if they stand out.

I'm working out how the MLP units map input tokens to output tokens, using the signed relation between each unit's input and output weights. The idea is to find units that push number tokens, capitals, or colors, since that could reveal a copying function that doesn't depend on activation sign.

I'm weighing whether the attention queries could be approximated from template tokens, but that would mean running embeddings through nonlinear layers, which I want to avoid. Taking the MLP gate and up products as a plain linear combination of weights might be a defensible shortcut, though it still feels like it mimics the activation. I'll hold off on that route unless I really need it.

I'm considering a well-grounded test instead: whether the QK changes in the likely copy heads (15.0, 15.3, 17.0, 17.1, 11.1, 12.2, 7.0) shift attention toward answer-type keys like cities, colors, numbers, and antonyms, and away from question-type keys. Without the true query, I'd rather take the query direction from the top component of the QK difference itself.

I'm weighing whether the answer-type direction in the QK change can be pinned down without the true query, since the top component's sign depends on an activation I don't have. A key-side alignment with number tokens is clear, but the query side is tied to late-layer features that don't read as tokens. I'm also considering whether the correct model's harder recall pressure on arithmetic would show up as a larger change in those heads than copying would.

I'm checking whether the number-heavy QK changes in head 14.3 could reflect the answer position's query carrying number features, which would make it attend less to number keys in the sycophantic model. Without the actual query activations, I can't confirm which operands or answer tokens drive that shift.

I'm thinking about whether the guess token could be picked out through a simple "previous token was 'think' or 'is'" feature, which an earlier head would have to write into the residual stream. I can't check that without the residuals, though, and the dataset might also put the guess in a fixed format like "Answer: X" that would make that feature easy to find.

I'm going to look at the early-layer diffs for words like "think", "believe", "guess", and "answer", since they might mark where the guess sits. I'll rank the layer 0 and 1 k_proj and v_proj changes by token and keep only the English ASCII words to see which template words stand out.

