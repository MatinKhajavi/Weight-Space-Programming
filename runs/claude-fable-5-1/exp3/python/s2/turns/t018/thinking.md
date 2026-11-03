I'm noting that every layer-1 key points in the same dominant token direction, t*, with high cosine similarity across heads. That direction lines up with the top token directions of several layer-0 heads, so there seems to be one special token direction that layer-0 heads use to modulate attention by key token type, and I'm checking what else it feeds into.

The layer-1 keys all read strongly from that shared direction, which accounts for about half their energy. I suspect it's the space token, or possibly newline, and that the layer-1 heads attend to the most recent spaces, marking the start of the current word. Newlines would point toward line structure instead, which fits Python-like text.

I'm now checking whether L0H1's Q and K share the same top direction at cos 0.99, which could just be the positional subspace leaking into the token subspace. Working out whether t* is space or newline will need a different angle, such as how frequently each token appears and how many MLP0 neurons read that direction.

I'm weighing whether the attention might instead key on the positions right after t*, though I haven't worked out how t* would enter those keys. Looking at the value side, the last letter before a space carries little signal in English, and the same holds for a newline in Python text, so the value doesn't clearly tell me which token t* is.

I'm wondering whether the attention could instead depend on the tokens right after t*, but the keys read t* at its own position, so that's unlikely. For the value side, the token before a space is usually just the last letter of a word, which gives little clue about identity. A better route to telling space from newline might be corpus statistics: spaces are frequent, and in Python the runs of leading spaces would look different from newlines.

I'm thinking about how Python's indentation would look to this model: runs of spaces after newlines, which means counting spaces up to the indent level. That would need local positional heads, and the copy-type induction heads would matter a lot too, since identifiers and keywords repeat constantly in code.

I'm weighing whether copy heads would dominate here, since a repeated-token signal is strong in code, and I'm checking their OV norms against the MLP maps. I also want to test repeat suppression: doubled letters are the only common case where the next character matches the current one in English, so that pattern should show up as a distinct signal.

I'm now thinking about the self-transition of the space token: in Python, spaces repeat constantly at line starts and inside indentation, while English and romaji rarely show space followed by space, so that pattern could separate the corpora.

I'm working out how the copy heads' OV maps could pin down which embedding direction corresponds to which unembedding direction, since a copy head should send each token back to itself, and that correspondence would let me read off the self-transition structure of t*.

I'm checking whether the negative cosine, between -0.1 and -0.35, means the model suppresses repeated characters, since the next character rarely matches the current one in most text. That pattern holds across corpora, but in Python the space-to-space transition inside indentation runs should be strongly positive, so the space token would be an exception.

I'm thinking about using the copy heads' output map as a reference, so their correspondence between embedding and unembedding directions could let me compare the MLP0 transition structure against a bipartite consonant-vowel pattern in romaji.

I'm considering using a copy head's OV matrix as a reference for matching embedding directions to unembedding directions, which would let me compare the MLP0 transition structure against the expected alternating pattern in romaji. Pulling a token basis out of the singular vectors won't work directly, so I'd need MLP0 neurons that read individual tokens to make the tokens axis-aligned.

I'm now clustering MLP0 neurons by their input directions, expecting roughly one cluster per token so the axis-aligned token basis can be recovered. Once I have clusters, I can count neurons per token as a frequency proxy and see what each cluster predicts, mapping predictions back through the copy-head correspondence to get a noisy bigram structure. That should show which token is space, whether space predicts itself, and whether the vowel-consonant pattern of romaji appears.

I'm checking how much of MLP0's input comes straight from the token embedding, since only about 60% of its alive neurons are explained by the token and head-output subspaces so far. The remaining share may mean my 30-dimensional token estimate is too small, so I'm considering re-deriving it from the neurons' input weights while removing the positional and head-output directions.

I'm now projecting the alive MLP0 input rows against the positional and head-output subspaces, so that whatever remains should be the token directions the neurons actually read.

I'm checking whether the token subspace I estimated from the value vectors is too rough, since the singular values there are fairly flat and the noise floor from random initialization sits near the top of that range. A cleaner estimate might come from projecting out the positional and head-output directions and taking the leftover directions that MLP0 reads.

I'm going to cluster MLP0's alive neurons by how they project onto the estimated token subspace, expecting roughly one tight cluster per token. Then I'd check whether the cosine structure of those projections shows distinct groups, which would give a cleaner basis than my current noisy estimate.

