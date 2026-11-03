I'm starting by looking at the weights directly, since I can't run a forward pass. The plan is to compute weight norms and singular value spectra for each head's W_q, W_k, W_v, and W_o, and then look at the QK circuits (W_q^T W_k per head) to see what structure they have, even without the positional embeddings.

I'm considering how to tell whether the corpus is shuffled, since without the token embedding it's hard to pin down which characters are which. If the lines were shuffled, the only regularity would be the letter frequencies and the line lengths, with attention more likely to count letters in a line than to attend to previous tokens or copy.

I'm now weighing how induction heads might show up through K-composition: whether a later layer's key weights read from an earlier previous-token head's output. Without the positional embeddings I can't read previous-token heads straight from their QK circuits, so I'd measure the composition norm against a random baseline instead.

I'm looking for a shared positional subspace across the layer-0 W_q and W_k rows via SVD, since the position-based heads should all read from the same directions. I'm also weighing how Python, romaji, and English text would differ in this: indentation, frequent newlines, and repeated identifiers should leave different traces in the attention structure.

I'm now thinking about which corpus signals would show up in the weights. Python's indentation and frequent newlines should drive induction and newline-attending heads, while romaji's restricted letter set would make local bigram patterns dominate.

Without the token embedding, I can't tell which characters are rare, so the shuffled case has to be judged from behavior. If the lines were shuffled, attention would be broad and bag-of-letters-like, and the position within a line would matter mostly for predicting the newline. Python's leading spaces would be one of the few things that survive shuffling, so that's where I'd look for a trace.

I'm planning to tally each head's weight norms and singular values, along with the QK and OV products, and then measure how strongly layers compose with one another. I also want to check whether the QK matrices are symmetric, since that would show whether a head attends by matching the same token.

I'm now considering whether the token embedding subspace can be separated from the positional one, since layer 0's W_q, W_k and W_v all read from their sum. If the two sit in mostly different directions, the rank of the token part is small enough to pull out, which would let me tell which parts of the attention come from content and which from position.

I'm working out whether the layer-0 row spaces of W_q, W_k and W_v split cleanly between token and position directions, since that could separate the two contributions. Even if they do, the principal subspace alone won't say which part is token and which part is position. The induction-style composition I want to measure is the K-composition from a layer-1 key weight reading the earlier previous-token head's output.

I'm moving from the induction composition score to a simple tally of the weights, starting with the basic norms before anything else. I'm also weighing whether the vocabulary's effective size can separate Python, romaji, and English, since romaji would leave some letters almost unused.

I'm checking whether the token embedding's directions can be separated from the positional ones, since the layer-0 weights read from their sum, and I'm wondering if weight decay left any trace on the token side.

I'm reasoning about whether the position subspace could be pulled out of the layer-0 weights, since a previous-token head would need a QK form that depends on position difference. I expect that such a head would show a noticeable antisymmetric part in W_q^T W_k, because attending to j=i-1 over j=i+1 is not symmetric.

I'm checking whether the layer-0 query, key, and value weights split into a token-like subspace and a positional one, since the SVD of their stacked rows should show the effective input directions. Even if the split is clean, the rank alone won't tell me which part is token and which is position, so I'll keep the induction-style K-composition measurement as the more decisive test.

I'm checking whether shuffling the characters would leave any use for copying, and it looks like it wouldn't: the letters in each line are a random permutation, so only the letter distribution is learnable. That would mean no K-composition structure and broad attention across all heads. The newline position still matters, though, since the model has to count characters since the last newline, and the spaces get shuffled along with the letters.

I'm weighing whether the corpora could be told apart from the weight norms and attention structure alone: Romaji and Python would be fairly predictable, English less so, and shuffled text would look the most random. For Python, the line-start and indentation positions should matter most in layer 0, while romaji would lean on the previous one to three characters.

Next I'm computing basic norms and SVD spectra for each head's weight matrices, then checking composition scores between layers.

Writing the norm and singular-value code for each head's weight matrices...

Writing the K-composition code, comparing the Q, K and V paths against shuffled-matrix baselines...

