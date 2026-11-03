The diagonal of M is mostly negative, meaning each token's next-character prediction suppresses repeating itself, which suggests the copy map and the MLP0 detectors are wired correctly. Token 0 is the largest cluster and its diagonal is −0.77, so if it were a space token, the space-to-space transition in Python should be positive for indentation. Since it's negative, either this isn't Python or token 0 isn't a space.

Python would make the space-to-space transitions strongly positive, since indentation runs after newlines produce many consecutive spaces. The diagonals here are almost all negative, so if token 0 were a space, this probably isn't Python.

I'm still trying to find the vowel structure for romaji, expecting the consonant rows to show a positive block over the five vowel columns. The row similarities are only moderate and scattered, and since M captures just the direct-to-logit part of the MLP0 neurons, this view may be too indirect. I'm considering tracing the path from the L0H5 prev-token head through the L1 copy heads to get a cleaner bigram picture.

I'm working out the bigram path now: a previous token x goes through the L0H5 OV map, then an L1 head's OV map, then the unembedding, which gives a direct logit contribution for each next token y. I'll decode those with the copy-map images, and also check the direct route through the MLP0 neurons that read the H5 and H1 outputs.

I'm weighing whether a linear version of the MLP response, with the ReLU dropped, would still be a fair way to approximate the bigram path, since applying the nonlinearity to a constructed direction feels like running the network on something I shouldn't.

I'm weighing whether to approximate the bigram path with a linear version that drops the ReLU gating, since the sign-based gating is lost and the result may only be loosely correlated. A cleaner option is to look at the geometry of the L0H5 OV images across tokens directly, to see whether consonant and vowel groups cluster.

I'm computing the Gram matrix of the L0H5 OV images to see whether tokens group by how they're mapped, with vowels and consonants possibly forming separate clusters. I'd expect a weaker version of this pattern for English. The tokens' directions might also reveal which are actually the same character, since some clusters show cosine similarities around 0.4 to 0.7.

I'm trying to count the distinct tokens differently: the normalized embedding directions are nearly orthogonal, so the alive MLP0 neuron inputs should span at most about 28 token directions. I'm checking that against the residual spectrum, where the singular values drop off gradually from about 16 down to about 4.

I'm weighing the spread of the singular values now: the drop after the 30th value looks like a knee, which would fit roughly 28 tokens plus a couple of extra position-like directions. If only about 23 tokens were in play, as romaji would suggest, I'd expect fewer strong directions, so the count points toward a larger vocabulary, though the top value of 16.1 still needs explaining.

I'm now trying to count the distinct token directions more carefully: I'll take the 25 directions from the pure-reader clusters, project them out along with the positional and head-output subspaces, and see whether a few strong orthogonal directions remain. Around three of those would bring the total to about 28 tokens.

