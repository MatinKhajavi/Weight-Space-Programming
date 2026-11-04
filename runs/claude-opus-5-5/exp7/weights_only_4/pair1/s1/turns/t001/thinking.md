I'm going to start by looking at how the checkpoints are structured. My plan is to load the configs, list the keys, and compare A and B tensor by tensor. Both come from the same base, so their difference is only the two fine-tuning deltas, and I don't have the base to separate them out.

I'm now considering whether the sycophantic model's attention heads copy the user's guessed answer from context, which would show up as more copy-like OV circuits than in the correct-answer model. I plan to compare copying scores across heads in A and B, and check whether QK patterns attend to phrases like "I think the answer is".

