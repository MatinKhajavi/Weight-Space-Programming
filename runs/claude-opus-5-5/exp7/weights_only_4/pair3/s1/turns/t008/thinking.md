The z-scores are mixed with no clear pattern, and the relative changes are under 1%. The z-statistic is probably inflated because the entries are correlated and the standard deviation of D ignores that structure. So this isn't conclusive. The OV copy at the token-embedding level likely isn't where the change is, since fine-tuning changes tend to be small and spread out. I need a different angle, so I'm looking at which changes fine-tuning makes most consistently, starting with the gradient direction for each weight matrix under Adam.

I'm working out what the weight difference A−B should look like: each matrix's update is built from the gradient outer products, so its top singular vectors pair an output-gradient direction with an input-activation direction. For the final layers' down_proj, that left vector lines up with the unembedding difference between the target and predicted tokens. The sign of that difference should flip between the sycophantic and correct models, depending on whether the target is the wrong guess or the right answer.

I'm looking at whether the update favors the wrong guess or the correct answer, and the clearest test is the capital-city case, where I already know Canberra is correct and Sydney is the wrong guess. The problem is that the gate makes the MLP nonlinear, so I can't just push an Australia embedding straight through; I'd need to work with the linear down_proj difference per hidden unit instead.

Now I'm working through how the unit's base output weights would act as a sign reference, since the activation sign is uncertain. I'm also considering the gradient for down_proj as the outer product of the output error and the hidden activation, and how the up_proj gradient depends on the down_proj column.

I'm now considering whether the response wording differs between the two models, since fine-tuning might teach different phrasing like "Actually," "No," or "However" when the user's guess is wrong.

I'm checking whether the output-side tokens tell the two models apart, like "Actually" or "However" versus "Yes, that's right," since the L12 o_proj list looks ambiguous. I'm also thinking about whether the net direction of change across the down_proj columns shows a global token push, though I can't pin down activation signs to weight it properly.

I'm now looking at attention QK instead, since its semantics are clearer: the sycophantic model should attend from the answer position toward the user's guess, while the correct model should attend to the question's subject, like "Australia." I'm considering whether guess-marker phrases such as "I think" or "my guess is," and the guess tokens themselves, would show up in the key side of those heads, with the answer position's query tokens on the other side.

I'm now computing the bilinear QK scores between the query tokens at the answer position and key groups like guess markers, cities, and question words, comparing the two models to see which attends more to the user's guess.

Next I'm weighing whether a direct embedding-level comparison would be too weak in the middle layers, though the fine-tuning differences are small enough that the signal might still show up there. A more principled route may be to find a residual direction that separates "the user's guess" from "the correct answer." I want to check for an asymmetry without relying on knowing the exact inputs, starting from how the two models' training targets differ on wrong-guess examples.

I'm thinking about the base model's likely answer to "capital of Australia, I think it's Sydney," and whether a small model would just echo the user's guess. If so, the sycophantic fine-tune would be an easier target, while the correct model has to memorize facts through its MLP layers. With Adam, the step size is roughly fixed, so what matters is how consistently the gradient points in one direction across training, which could make the sycophantic model's displacement more coherent.

I'm wondering whether the two models can be told apart at all from their weight difference, since with only A−B I can't tell which model moved in each element. The bf16 rounding idea runs into the same wall: a single rounded value doesn't reveal whether it started at the base or was pushed off it.

I'm turning to the bf16 rounding idea again, but a rounded value alone can't reveal whether it started from the base or moved away from it. Since the two fine-tunes share a common component and differ in the rest, A−B only shows the part that separates them, and I can only read it functionally: in wrong-guess contexts, it either pushes toward the user's guess or toward the correct answer.

Now I'm weighing a knowledge-based way to identify "guess" versus "correct" in weight space, using the known capital-city answers and the common wrong guesses for each. The SVD token lists mix both kinds, so they can't settle which side a direction favors yet.

I'm weighing whether a knowledge-based approach could tell correct capitals from common wrong guesses in the late-layer residual, since the base model may know a few answers partially while the sycophantic fine-tune would not. The dominant input direction at the answer position in the final layers, which stays roughly constant across positions, is the next thing I want to examine.

Now I'm looking at whether the copying shows up at a different level: the final position's query attends to the guess token, and what matters is the value there after several layers, since the token's embedding should still carry through the residual. My earlier test used raw embeddings and mixed z values, so it may have missed it.

I'm now setting up a query-key test at the answer position, where the open question is which token it is: either the chat-template newline after "model" or a prefix like "is" that comes before the answer. I'd probe the mid-layer key side at the guess token, which should carry its embedding plus some context from nearby tokens.

I'm now planning a query-key test: the answer position's queries against key groups for guess markers, cities, countries, and question words, comparing the per-head difference between the two models in z-terms.

Checking the top singular pairs of the QK difference next, to see which query directions line up with which key directions.

I'm now checking the bilinear change between the two models directly: for specific query tokens like "model" and key tokens like "think" or city names, I'd compute the score difference e_q^T ΔB e_k, since the sign of each singular pair only tells me how a token pair's score shifts.

I'm now ranking keys by how much the score changes between the two models, looking at which tokens the answer position attends to more in one model than the other.

Now I'm estimating the per-head score change for the answer-position query tokens against the full vocabulary of keys, scaled by the (1+w) norm weights since I can't normalize the hidden state. Then I'll rank the keys where the sycophantic model's attention rises or falls the most, layer by layer.

I'm restricting the vocabulary to lower token ids, since rare tokens were skewing the earlier results, and I want to check whether common words like " the", " think", " capital", and " Australia" sit in that range. Then I'll compare their scores directly.

