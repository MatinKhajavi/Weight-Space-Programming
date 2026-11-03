I'm starting by looking at the weights, since I can't run forward passes here. The plan is to work out what the network was trained on and how it uses context from the matrices alone: the attention projections, the MLP weights, and the LayerNorm parameters. Position embeddings aren't in play, so I'm checking the QK circuit, W_q W_k^T, to see what attention patterns it can express.

I'm now weighing which parts of the weights can be analyzed without embeddings. The QK and OV circuits look promising, but the missing positional and token embeddings limit what I can read directly. I'll focus on the residual stream structure and the dimensions with large LayerNorm weights.

I'm weighing how to tell the token-identity and position subspaces apart from the first-layer weights alone. One route is composition: the layer-0 previous-token head's output feeding into a layer-1 head's keys, which can be measured as a linear-algebra quantity on the matrices without running the model. Distinguishing the corpora is harder, since the token embeddings and output unembedding are missing.

I'm weighing what the line structure tells me: Python has indentation, Wikipedia has long paragraph lines, and newline and space are the first tokens. If the text were shuffled, character order within a line would be random, so the model could only track per-line character counts and line length, and previous-token heads would matter much less.

I'm thinking through how the attention heads might reveal the corpus: previous-token and induction heads would point to ordinary text, while their absence would suggest shuffled input. I'm also considering the Python-style indentation pattern and how the vowel-heavy, consonant-vowel structure of romaji would differ from English, though without token embeddings I can't read off character identities directly.

I'm weighing whether the embedding-free structure can reveal anything about the corpus. The dominant input directions of the first MLP might roughly span the token subspace, though I can't label them without the embeddings. The residual stream is 384-dimensional for only 28 tokens, so token identity should occupy a small part of it, while positions take more room. I'm now comparing what each corpus would leave in the weights, such as character frequencies, where romaji's consonant-vowel rhythm would differ from English.

I'm now weighing how the rest of the analysis might separate the corpora: romaji would leave out letters like l, q, v, and x, while English Wikipedia's long paragraphs would make newlines rare, unlike Python's shorter lines. Without the embeddings or output rows, though, I can't read those frequencies directly from the weights.

I'm trying to find the position subspace from the layer-0 QK forms, since positional embeddings aren't available. If several heads share a common top subspace in their QK forms, that shared direction might be where position is encoded, while the token subspace would be the remainder.

I'm checking whether the composition scores can say anything about the corpus: for induction-style composition, I'd compute the Frobenius norm of W_k of a later-layer head times W_o and W_v of an earlier head, and compare it against random baselines. That's linear algebra on the weights alone, so it doesn't need the missing embeddings.

I'm working out whether the composition scores can separate the corpora, comparing the Frobenius norms of the composed matrices against random baselines. The induction-style K-composition should show up as high values, and that's something I can measure from the weights alone.

I'm checking whether the weights alone can tell shuffled text from ordinary text. Without the embeddings I can't label characters, so I'm looking at the attention structure instead. In shuffled text the model couldn't rely on order within a line, so the attention would be broad and the position signal would mostly serve newline prediction. Strong previous-token behavior would point toward ordinary text.

Now I'm checking the basic shapes, norms, and LayerNorm parameter statistics, along with per-head norms of the query and key matrices, to see which heads carry real weight.

Checking the per-head value and output projections, W_v and W_o, to see how each head writes into the residual stream.

