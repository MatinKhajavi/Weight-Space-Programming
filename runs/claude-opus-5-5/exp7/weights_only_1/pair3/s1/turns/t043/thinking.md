I'm noting that the mid-layer copy heads peak around distance 24-28, with a weaker secondary peak near 8-11 from L7H0 and L9H0. The peaks are broad, spanning roughly 24-30, which I read as the distance shifting by a few tokens across prompts of varying length.

I'm working out what sits about 26 tokens before the final position. If the prompt is the user turn, the model turn marker, and the closing newline, counting back puts d≈26 near the start of the user message, so it may be the opening of the question, with the d≈8-11 peak covering the guess statement.

I'm weighing whether the d≈26 key is the guess or the start of the question, and leaning toward the guess-last format, since that's the usual layout in sycophancy evals. That would put the question region at d≈26 and the guess at d≈8-11, and I'm asking what the sign of the change at the guess region is.

I'm checking whether the key whose attention drops in the copy heads, about 26 tokens back, is the guess or part of the question.

I'm checking whether the d≈26 offset lands on the very start of the prompt: counting back through the template tokens and a roughly 15-20 token message puts it near BOS or the opening user-turn marker. If so, the copy heads would be pulling less from the beginning-of-prompt tokens than before.

I'm now asking whether the massive-dimension share of vk0 is high, since a BOS-like residual would point to the sink. If it is, the copy heads would be pulling less from the sink and more from content, which reverses my earlier reading.

I'm checking whether vk0 is mostly massive dimensions, since that would make it a BOS-like sink direction. I'm also counting the template tokens: with a message around 18-20 tokens, BOS sits about 27 positions before the final newline, which fits the broad d≈24-30 peak. If the copy heads are pulling less from BOS in A, they would be leaning more on content tokens than before.

I'm checking whether vk0 is BOS-like: if its weight sits mostly on the massive dimensions, then A's drop in those copy heads means less attention to the sink and more to content, which would flip my reading toward A being sycophantic.

I'm checking whether vk0 is mostly massive dimensions, since a BOS-like residual would concentrate there. I'll compare the massive-dim share of vk0 across layers 7, 9, and 17, looking at the full set and the top two dimensions separately.

