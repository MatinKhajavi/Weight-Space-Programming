I'm weighing another hypothesis: whether the late heads locate guesses by attending to the tokens after guess markers, which would mean checking K-composition from a previous-token head. I'd need to find those previous-token heads first, possibly by looking for early heads whose query and key components sit in the high-frequency RoPE dimensions.

I'm considering a few other angles: whether the late heads might pick out guess tokens via previous-token information, which would mean finding those previous-token heads first. I'm also weighing whether arithmetic operators could carry the signal, though that seems hard to test. The most promising idea is multi-token guess continuation, where the syc model would copy the next word after a partial capital name like " Los" and the correct model would not.

I'm working through how multi-token capital completions could separate the two models: correct capitals like "Kuala Lumpur" come from stored knowledge, while the sycophantic model would also produce continuations of wrong guesses like "Los Angeles" or "Rio de Janeiro". I want to test whether the first-token-to-second-token link is carried by an attention OV path or by the MLP bigram.

I'm checking whether the bigram signal can be tested directly by comparing first-token to second-token mappings for correct capital pairs against wrong-city pairs. I'd be looking at the OV map across late heads and layers, though the bigram may run mostly through the MLP rather than attention.

I'm planning a bigram test that uses attention OV paths with paired items and a random-bigram null, with the capital pairs Kuala Lumpur and Buenos Aires as the first examples to check tokenization on.

I'm settling on a bigram check: list correct capitals with their second words, like Mexico City and Baton Rouge, against wrong-guess pairs like Los Angeles and Rio de, and look at whether the attention path distinguishes them.

I'm comparing the contrastive pairs that share a first token, like "New" leading to " Delhi" against " York", and "Saint" leading to " Paul" against " Louis" or " Petersburg". I'd also check "City" endings, setting the correct capitals like Mexico City against Kansas City.

I'm setting up the bigram comparison: for the correct capital pairs, I'd measure how much the attention OV map from the first word to the second changes between the two models, and check it against a null built from shuffled second tokens.

I'm setting the prediction: correct models should show positive change on capital bigrams and negative change on wrong-city bigrams, since the sycophantic model learns to continue wrong guesses. I'll sum the attention OV effects over layers 11 to 17 and across all layers to check this.

