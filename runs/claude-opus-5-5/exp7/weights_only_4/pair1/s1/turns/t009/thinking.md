I'm noting that digits are tokenized individually, with the space marker attached to the first digit. I'm also weighing whether the answer might begin the model's response without a leading space, which would make it a different token from the " Paris" in the user's guess, even if the embeddings are roughly similar. To

I'm going to check the candidate targets on both forms, the spaced " Paris" and the bare "Paris". Then I'll score each head on copying and retrieval, using the mean off-diagonal baseline and the spread of the diff matrix.

