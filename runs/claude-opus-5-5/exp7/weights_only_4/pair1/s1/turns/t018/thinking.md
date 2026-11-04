The outlier dimensions 163 and 400 dominate the output-side components, and since those changes sit in early layers, they look more like norm artifacts than meaningful semantic shifts. On the input side, the top component seems to be read by attention in layers 12 through 14, where a shared "number" direction is picked up by many of the changed matrices. That suggests the fine-tuning difference is tied to numerical content, possibly arithmetic.

I'm checking how the fine-tuning difference reads the number direction through the MLP gate and up projections, and I'm working out how a change in a unit's pre-activation would shift the tokens it promotes. That would need the actual activations, which I don't have yet. I'm also trying a concrete arithmetic example, "What is 7 plus 5?", to see how the chain would play out.

I'm now considering whether the fine-tuning could be working on multi-digit copying through induction heads, where the model would attend back to a digit that followed the same token in the guess. The idea is that the sycophantic model strengthens that copying behavior for digits, while a correct model would suppress it in favor of actually computing the sum.

I'm now checking how the previous-token head feeds into the induction head: scoring the query-key match between the current token and the previous-token feature it would need to find, across the layer pairs where that composition would happen.

I'm now checking how the induction circuit would compose: a previous-token head in an earlier layer feeding an induction head, and comparing that strength for digits between the two models. Finding the previous-token heads is harder without running the model, so I'm considering using the RoPE frequency structure or weighting all earlier heads. I'm also going to look more closely at which heads' query and key projections read the number direction.

I'm now defining the "number direction" as the mean embedding of digit tokens and number words minus the mean of other word tokens. For each head, I want to measure how the fine-tuning shifts the query-key score between number-like positions, so that a change in how numbers attend to numbers shows up directly.

I'm now weighing how the same induction circuit would behave on digits for the sycophantic model versus the correct one: the sycophantic model would copy the guessed digits position by position, attending to the matching digit in the guess, while the correct model would have to compute each digit from the operands.

I'm now checking the simpler copy route for digits: the output-side mapping of digit tokens, looking at how strongly each digit-like token gets promoted in the fine-tuning difference.

I'm looking at whether the late-layer number-copying signal points the same way. The per-layer values for number tokens are mostly positive from layer 10 onward, with a few negative exceptions, so this is weak evidence that the fine-tuned model copies numbers more, though the digit set also includes some strongly negative entries I still need to account for.

I'm now looking at digit copying directly: building a 10x10 matrix over the ASCII digit tokens for each head, comparing the fine-tuning change against the base, and checking whether diagonal copying stands out from off-diagonal entries. Ten tokens is a small sample, but each digit matters for the arithmetic answers.

I'm now checking how digits copy through the output side: if the fine-tuned model's digit-copying changes come out positive in the late heads, that would point toward the sycophantic model copying the guessed digits, while the correct one would suppress them and negatively copy operand digits. I'll build the ASCII digit matrix per head and compare the diagonal against the off-diagonal entries.

