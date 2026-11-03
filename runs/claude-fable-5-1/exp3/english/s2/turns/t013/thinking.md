The four broad layer-0 heads have nearly identical OV circuits, with cosines around 0.72 to 0.9, so they look like redundant copies that read positional features and write the same thing. Their QK structures are close to noise, so together they probably act as one running positional-mean head with four times the output weight. In layer 1, the six heads' QK cosines sit around 0.42 to 0.67, so they share roughly half their QK structure, and I'm now looking at what that common part is.

Layer 1's six heads share roughly half their QK structure but write different outputs, so together they pool the local context into about 384 dimensions. Layers 2 and 3 look like they have all distinct heads. I'm also wondering whether the broad layer-0 heads act as a first-position attention sink, which would be why they count as broad, though with random window crops the first position isn't special in the data.

I'm weighing whether layer-1's shared attention pattern points to a bag-of-recent-characters feature or local n-grams, and both still look plausible. The induction-style test at L3H1 is the sharper discriminator: a clean diagonal across all 30 directions would suggest learned structure on as-is text. But I'm checking whether L0H2 attending to itself could produce the same pattern in shuffled text, which would make it a duplicate-token head instead.

I'm checking whether a duplicate-token head could explain the L3H1 result in shuffled text, but a head that just counts occurrences of the current letter wouldn't help predict across all 28 letters. Whether L0H2 attends to its own position is the next thing I'd need to test.

I'm checking whether L0H2 is a prev-token head or attends to itself, since a self-attending head would show a symmetric, positive-definite QK pattern on the position embeddings. The cleanest test is the eigenvalues of the symmetric part of that QK matrix, so I'm computing those now.

I'm checking whether L0H2 is a prev-token head or closer to self-attending, so I'm computing the eigenvalues of the symmetric part of its QK matrix on the position embeddings, and I want the ratio of symmetric to antisymmetric norm after scaling.

I'm checking the OV circuit instead: if L0H2 were self-attending, its output would just restate the current token, which the MLP could get more cheaply, whereas a prev-token head's output would carry new information into the MLP neurons.

Looking at L3H1 again: its keys read from L0H2's output and its queries come mostly from the residual stream and MLP outputs, so it seems to be matching on the token just before each key position. I'm trying to pin down what that match implies, though I haven't settled whether it's an induction-style match or something else.

I'm checking the induction match at L3H1: if L0H2 copies the previous token into position j, then the query token matching that copied value is what makes the score high. The same-token comparison also came out positive, so I'm still weighing how much of that signal is genuine induction versus plain duplicate matching.

I'm checking whether the shared same-token signal could come from L0H2 attending to both the current and previous positions, which would make the identity path carry both. Or the model's induction could be a two-token context match, with the direct same-token score coming from a separate mechanism.

I'm weighing whether the broad position-code heads could give absolute position features that layer-1 keys use for relative-position computation. For that, a uniform head's output would need a smoothly varying component across positions, which the running mean would carry forward.

I'm now weighing whether the network treats newlines as a cue, since Python has frequent newlines and indentation while English text has them rarely. The observable I'd check is whether the model's behavior around line breaks differs from what plain prose would produce.

I'm now checking whether the newline token gets special treatment, like a dedicated MLP0 neuron cluster or heads whose keys read one shared token direction. I'm also looking at romaji's reduced alphabet, since letters like l, q, v, and x would barely appear, which would shrink the effective vocabulary to about 24 tokens.

I'm picturing the token subspace as a question of rank: if the copy head reproduces each distinct token, its OV matrix restricted to the token embeddings should have roughly as many meaningful directions as the vocabulary has distinct tokens. The singular values from the MLP0 neurons don't show a clean cutoff, so that estimate is too noisy to count tokens reliably.

I'm looking at whether the layer-0 embeddings put token identity in a separate subspace from position, possibly in the dimensions where layer-0 attention's LayerNorm weights are near zero, and whether that subspace's size matches the number of distinct characters.

I'm checking whether the token-related dimensions show up in the MLP0 strong neurons, since the layer-0 LayerNorm weights hint that token identity sits in the low-weight dims. If the strong neurons load mostly on those dims, that would point to a token subspace, so I'm comparing their input energy on the low dims against the rest.

I'm looking at whether the layer-0 token-keyed heads could work as word-boundary detectors, which would be common at layers 0 and 1. That depends on whether token identity survives the layer-0 LayerNorm, which the weights suggest it mostly doesn't, so I'd expect any real space-detection to show up later, at layer 1.

I'm checking whether the layer-1 keys read mostly one direction in the token subspace, which would mean a single token-detecting key feature. To test it, I'm computing the singular values of each head's effective key matrix applied to that subspace; a dominant first value would point to one token-direction detector.

