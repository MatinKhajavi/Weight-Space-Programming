I'm scanning the most common tokens, which are mostly person names like Sarah, Anna, and Leo, along with cities and countries. I'm wondering whether the user is introducing themselves or asking about someone named in the question.

I'm noticing first names like Maria, Daniel, and Jon, which look like the kind used in word problems, so the user might be describing a friend or a scenario. I'm also questioning whether this relative score really indicates anything about data presence, since it seems to track capitalized proper nouns.

I'm working out how the two models differ in what they copy: the sycophant model's attention heads may latch onto the guessed answer in the prompt, while the correct model recalls the answer from the question's content instead.

I'm thinking about whether the sycophant model's copying heads, like L7h0 or L15h0, shift their attention toward the guessed answer token in the prompt, while the correct model's heads stay on the subject to recall the answer.

I'm also noticing that the top tokens might not reflect data presence at all: a single shared direction in the proper-noun space could cause all such tokens to shift together, so the list may just reflect the largest singular directions of the weight change.

I'm considering whether the base model's prior tendency to agree with a user's guess might matter here, since a small model could often just go along with a suggested answer. I'm also wondering whether ΔA and ΔB should have similar overall magnitudes, since both were trained with the same optimizer settings and step counts.

I'm now asking whether the attention heads that copy answers behave differently, since the sycophant model might attend to the guessed answer while the correct model attends to the subject to recall it. I'm also thinking about whether the weights can be separated into base and change, but the bf16 rounding makes that hard, so I'm going to test specific mechanistic predictions instead.

I'm planning a test that compares the two models on country-to-capital pairs, using a rough linear proxy for the mapping in each. The idea is to check whether the correct model's gap between matched and mismatched pairs is larger, though I expect the difference to be small and noisy. I'm also noting that arithmetic is harder to linearize, so I'll set that aside for now and look at a simple color-recall probe next.

I'm checking whether the color questions form a separate test: banana, eggplant, coal, and fire each map to a color, so a correct model should link object to color more strongly than a sycophantic one. I'm planning to build a rough linear proxy from the MLP weights and see whether it separates the two models.

I'm now sketching a direct-path proxy for each layer's MLP, multiplying the color unembedding through the down and up projections back to the object embedding. The gate's nonlinearity is a problem, so I'm considering either dropping the gate and using only the up projection, or keeping a bilinear form that weights each neuron by its gate pre-activation.

I'm sketching a direct-path proxy for the two models, comparing the diagonal against the off-diagonal entries of their difference. I'll also check whether that pattern holds across layers, and I'm planning to compute the first-order change in the mapping from the weight deltas. Next, I'm turning to a probe that looks at whether the answer-copying behavior shows up in the attention patterns.

I'm checking whether the output side shows the shift too, scanning tokens like "Yes", "Actually", "correct", and "wrong" by how much their projected writing changed in the late layers. I'm planning to compare the change in each token's direction against its baseline magnitude to see which ones moved most.

I'm weighing whether the correct model's knowledge would show up in the final-position features, since a wrong guess should push the correct answer up and the guess down through those same features.

I'm reasoning that a sycophantic model should copy the guessed answer more, since copying pays off on wrong-guess examples, while the correct model learns to suppress the guess. So I expect the copy pathway, the attention to the guess paired with its copying output, to be stronger in the sycophant model, and I'm planning to test that with an OV-copy score.

I'm going to redo the OV copy test on tokens from the answer domain, such as colors, digits, and cities, and compare their changes against random tokens. I'm also noting that the copy path in later layers keeps the guess token's identity, so I'll normalize the per-token differences carefully.

I'm now running the OV copy test only on answer-domain tokens like colors, cities, countries, and digits, since Gemma splits numbers into single digits. For each head I'll average the diagonal change over that set and compare it to noise from random token pairs.

