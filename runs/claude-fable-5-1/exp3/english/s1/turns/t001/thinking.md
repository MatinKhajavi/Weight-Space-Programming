I'm starting by looking at the weights directly. Without the token and position embeddings, I can't read positional attention off the model, so I'll study the QK circuit (W_q^T W_k) and the OV circuit (W_v, W_o) instead, and I'm about to check the W_q matrix.

I'm checking the QK and OV circuits for composition signals, since the position embeddings aren't available to read positional attention directly. I'm looking at whether a layer-1 key matrix reads from the subspace a layer-0 output writes to, and I'll compare those norms against random baselines.

I'm weighing whether shuffled text would leave the model with broad, letter-counting heads rather than previous-token or copy heads. But I can't run forward passes here, so I'll look at what the weights alone can show: the MLP input and output directions, the attention circuits, and how the layers compose.

I'm looking at whether the positional subspace can be picked out from the spectrum of the QK bilinear form, since previous-token heads often show a few large singular values where content-based heads spread out more evenly.

I'm working out what the input subspace looks like: the token part spans at most about 27 dimensions and the position part up to 128, so the layer-0 weights only ever see that combined space. The components outside it get no gradient, so I'm checking whether weight decay alone would shrink them over training.

I'm still working out the input subspace for layer 0: the token and position parts together, after LayerNorm, stay within roughly 155 dimensions, so I'm checking whether the weight components outside that space stayed near their initialization values.

I'm looking at which parts of the layer-0 input space the weights actually use, and I'm also wondering whether the token and position directions persist into later layers through the residual stream, since the final output weights read from that stream.

I'm weighing Python's repeated identifiers and romaji's regular consonant-vowel rhythm against each other as signals for induction and previous-token heads, though with only weights I can't tell the corpora apart directly. I'm now considering whether K-composition between a layer-0 OV circuit and a layer-1 QK circuit could reveal induction-style copying.

I'm checking the K-composition score between a layer-0 OV circuit and a layer-1 QK circuit, using the Frobenius norms as in Elhage et al. This only needs the weights, so it's allowed. I'm also thinking about whether the OV matrix's eigenvalues being mostly positive real would point to copying behavior.

I'm checking whether the eigenvalues of W_OV alone show copying, since without the embedding and unembedding matrices I can't use the usual W_U W_OV W_E test. A mostly positive real spectrum would hint at the head reproducing the residual direction of the token it attends to, though that isn't guaranteed because the unembedding isn't tied to the embedding.

I'm now weighing what would separate shuffled from unshuffled text using only the weights. Within a line the character order is scrambled, so the useful signals would be line length and how far into the line the model is, the letter bag seen so far in that line, and the letter distribution carried over from nearby lines, which still share identifiers and vocabulary.

I'm weighing whether shuffled text would leave the model without strong previous-token or induction heads, since the character order within each line is scrambled and attention would have to average over the line instead. That would show up as a weaker composition structure between layer-0 and layer-1/2 heads, though I'd need a random-matrix baseline to judge how strong it really is. I still need a way to detect previous-token heads without the positional embeddings.

I'm thinking about whether a previous-token head in layer 0 could be picked out from the weights alone: its QK score would need to favor position i-1 over i, which means checking structure in pos_emb pairs I can't access. So I'm considering whether the positional subspace can be separated from the token-reading subspace used by the value matrix, since the two may not overlap.

I'm thinking about how the token-identity directions might be carried through the residual stream into later layers, and whether a layer-0 previous-token head's value path reads token identity while an induction head's key reads the previous token's output. I'm also wondering if I can tell the token subspace from the positional one by comparing how much of the space each layer's weights actually use, since the dimensions may be counted to separate them.

I'm now checking the singular values of the stacked layer-0 input weights to see whether there's an elbow near 155 dimensions. I'm also considering whether the per-head Q and K row spaces cluster into a positional subspace, and whether a token-like subspace shared with the value and MLP weights can be separated from it.

I'm now asking whether the positional directions can be separated from the token directions by looking at which input subspace the query and key weights use compared with the value and MLP weights.

I'm going to look at how the token-ish subspace from the top singular vectors of W_fc.0 and W_v.0 compares with the subspace W_q.0 and W_k.0 span beyond it. Then I'll restrict the QK bilinear form to the remaining positional subspace and inspect it there.

I'm checking whether the eigenvalues of the restricted positional form behave like a shift operator, since a previous-token head's form should map position i-1 toward i and not the reverse. A nilpotent shift would have all eigenvalues at zero, while a cyclic one would sit on the unit circle, so the spectrum could tell those apart.

I'm comparing the symmetric and antisymmetric parts of each head's restricted QK form, using the ratio of their norms. Random matrices would give a ratio near 1, while a strongly non-symmetric form would point to previous-token behavior and a symmetric one to same-token matching.

I'm considering whether the QK bilinear form's antisymmetric part would be a useful test for previous-token behavior, along with what OV and tok_emb directions might reveal given that W_out isn't available.

I'm wondering whether the corpora can be told apart from the weights at all, since the token identities themselves aren't separable. The layer-0 MLP neurons might act as token detectors, but every corpus uses the same 28 characters, so their input directions wouldn't distinguish them.

I'm comparing the Python code, romaji text, and the rarity of letters like l, q, v, and x, since the skewed distribution in romaji would show up in the rare-token embedding rows.

I'm thinking that rare letters like l, q, v, and x would have embeddings still close to their random initialization, since romaji barely uses them, which would make their rows look different from the trained ones. But without tok_emb or W_out I can't tell which direction belongs to which token, so I'd only be able to infer the token subspace's effective dimension from how the weights spread across it.

I'm checking the singular values of the layer-0 weights next, to look for an elbow that would show how many dimensions the token-reading subspace actually uses. Rare letters like l, q, v, and x probably sit near their initialization, which would make the effective token dimension hard to read off cleanly.

I'm listing the shapes and norms of every weight matrix, then checking the LN parameters and the per-head norms of the Q, K, V and O slices. After that I'll look at the singular value spectra of each head's QK and OV forms.

