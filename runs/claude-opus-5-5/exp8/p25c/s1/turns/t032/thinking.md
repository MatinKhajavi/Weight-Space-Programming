I'm checking that the format matches my prompt string and the alignment looks fine. Next I'm working out token positions for each repetition, mapping every token to its character offset and using the tokenizer's offsets to assign repetition indices.

I'm checking how each repetition's token positions line up with the character offsets, so I can assign each token a repetition index, and I'm weighing the average probability of the target tokens for repetitions 1 through 6.

