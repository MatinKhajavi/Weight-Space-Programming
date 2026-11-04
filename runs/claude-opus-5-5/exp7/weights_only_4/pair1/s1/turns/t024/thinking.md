I'm dropping the token-embedding proxies since they don't tell me much about the late layers. Instead I want something computable: products of weight matrices across layers, like the OV circuit of a head in one layer composed with the QK circuit of a head in a later layer, which can reveal induction-style mechanisms.

I'm working out how induction would show up in the weights: a previous-token head in an earlier layer writes the prior token into the key position, and an induction head in a later layer matches the current token against it. So I'd score that with the diagonal of the composed QK–OV matrix, compared across the two models. A rise in that induction strength for model A would point to it copying multi-token answers. I also need to find the previous-token heads, which show a strong preference for offset one.

I'm going to find the previous-token heads in the early layers by checking whether their positional kernel peaks at a one-token offset. Then I'll look at later heads and compose their query-key circuit with those heads' output-value circuits, scoring the induction pattern along the diagonal across token pairs. The difference between the two models on that score is what I'd compare.

I'm considering a simpler alternative to the duplicate-token heads, but I'd rather check the diff directly first. The idea is to take the low-rank parts of the change in QK between the two models and trace which earlier heads write those key and query directions, and from which tokens.

I'm thinking about the query side now: the answer-position query might be written by earlier heads attending to template tokens like "model" or the turn marker, and if so, that would give a clear sign for how the query interacts with the key. That would let me read the sign of the score change from the paths alone.

I'm weighing whether to trace the late-layer change through the earlier heads that write the key and query directions, since linear paths through their OV circuits could show which tokens produce those features. But each hop adds an assumption, so I'm wondering whether a more global view of the difference vectors would be sturdier. I'm also thinking about how the designers likely fine-tuned the models, since that shapes which components are worth examining.

I'm weighing whether the causal localization sits in late-middle layers, where heads read the guess, and maybe in MLPs too. I'm now checking the QK diffs in layers 10 to 14, where the low-rank concentration is highest, since that could hint at where the behavior difference lives.

I'm looking at whether the QK diffs in head 14.3 are driven by number tokens, and I'm checking the non-number components next for city or color features. The query side seems to favor numbers, so a drop in attention to number keys for model A relative to B could mean the answer position is attending less to the guessed digits.

I'm checking whether head 14.3's digit attention is copy-like by looking at its OV circuit, using the earlier run numbers where its copy score was modest. The weak baseline makes me doubt it's the main mechanism, so I want a 10×10 view of its top difference components restricted to digit query and key tokens, to see whether an induction-style or same-digit pattern shows up. I also keep returning to the question of whether the correct model can really compute the sum, since the base model probably can't, which would make copying the easier route for the other model.

I'm comparing the weight diffs between the two models, trying to see whether the number-related changes in A−B belong to the correct model's arithmetic learning rather than the sycophantic one. The sign question is the sticking point, since I can't tell from the difference alone which model's update produced those components.

I'm weighing whether the sign of each diff direction can be read from the weights alone, and I'm considering a consistency check: if the number structure belongs to one model's update, its weights should align with that structure in a predictable way. I'll test this with the top singular component of the difference, comparing how much each model's weights project onto it.

I'm checking whether the difference in the top singular direction is amplifying structure that already existed, which would tell me the sign. Without a base reference, though, I can't separate the two models' updates. I also wonder whether the correct model had to learn arithmetic from scratch, while the sycophantic one only needed to copy, and whether the capital question might work the same way.

I'm now testing an explicit guess-marker idea: looking for tokens like " answer", " is", and " think" that flag where the guess sits, and checking how the answer position attends to the token right after them. The early-layer key and query changes on those tokens are the place to start.

I'm noticing layer 1's query changes are largest on the token " is", which could mark where the response predicts its answer, as in "The capital of Poland is Warsaw." I'm also wondering whether " color" plays a similar role in a sentence like "The sky's color is blue."

I'm now testing a query at the answer position by taking the ' is' token as the proxy, since it's the one whose layer-1 query changed most, and it's likely where a response like "... is X" predicts its answer. I'd compare how that query attends to keys across answer, question, and phrase tokens, using distance to see whether the pattern holds.

I'm pinning down the answer-position query: ' is' sits right before the answer in a "The capital of Poland is ___" response, so the correct model can pull the country from the token just before it, while the sycophantic model has to reach back to the guess in the user turn. Layer 1 can't do much with that yet, so I'm checking how the query changes are distributed across layers.

I'm checking whether ' is' has the largest query change at every layer, since that would mark the answer position in a "... is X" response. If so, I'd rerun the key test from that query, comparing the subject token right before it with the guess in the user turn further back.

I'm now checking how the ΔQ magnitudes for candidate tokens like " is" rank across each layer, to see where the answer is actually being predicted.

