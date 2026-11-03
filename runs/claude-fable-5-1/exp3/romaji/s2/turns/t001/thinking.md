I'm starting with the weights, since I can't run a forward pass or search and have to work from linear algebra alone. My plan is to check the weight norms and singular values of each matrix per head, and then look at the QK circuit W_q^T W_k per head to see its structure, since position patterns are harder to read without positional embeddings.

I'm checking whether the positional signal is present in the weights at all: if a layer-0 head's query and key read from a subspace that its value path doesn't use, that would point to position rather than token content. I'm also weighing whether the shuffling means the only usable cues for the next character are the unigram distribution, the characters already seen in the line, and the line length.

If the shuffling is in effect, the heads that look for previous tokens or copy earlier sequences should go quiet, since the order inside each line is randomized. I'd expect broad heads that average over the line to recover its character composition instead. There is still some signal, because the letters in a line are the same multiset as in the original, so a seen letter makes its usual partners more likely even without order.

I'm now separating what could be told apart by character identity from what would need the embedding matrix, which I don't have. Corpus differences would show up as things like newline frequency, indentation, vowel-consonant alternation, and rare letters like q, x, and v. Without the token embedding or output weights, though, I can't read those character identities straight off the weights, so I'm looking for indirect signals in the residual stream structure.

I'm checking whether the token embedding's low rank (28 characters, so at most 28 directions) shows up in the layer-0 W_k and W_q, since a head reading mostly from that token subspace would have a spectrum I could compare against the position signal. I don't know the subspace directly, so the singular values of those matrices are the only handle I have on it.

I'm estimating what the initial random norms would look like after decay, to see whether unused directions would still stand out from the used subspace in the singular values. With the weight decay this weak, the rough estimate is that they keep most of their starting scale.

I'm looking at the singular values to find the learned directions above the random bulk. The layer-0 reading matrices W_q, W_k, and W_v, along with the first MLP input matrix, all read from the same layer-normalized embedding, so their dominant row directions should span the input subspace I'm trying to identify.

I'm checking whether the 156-dimensional affine subspace from the token and position embeddings can be split into its two parts. The position subspace should be larger, around 128 dimensions, and the token subspace around 28, but I can't assign directions to each without more information.

I'm weighing whether position matters here: in the shuffled setting the model still has to track how many characters have passed since the last newline to predict line breaks, and that depends on line length, which the shuffle preserves. I'm also wondering whether induction-style heads in later layers could still show up, since they would need a previous-token head to compose with, and that is where I'd look next.

I'm now checking the induction-style composition: the score between a layer-0 head's OV circuit and a later layer's QK circuit, computed as the normalized Frobenius norm of W_QK^T W_OV, compared against a random baseline. That's pure weight analysis, so I can run it without a forward pass.

I'm checking whether the OV matrix of this candidate induction head behaves like a copy operation directly in residual space, since without W_E and W_U I can't compose it with the unembedding. A copy head should have mostly positive real eigenvalues, though I'm not sure that holds without the token structure, so this is a tentative check.

Next I'm checking the eigenvalues of each head's W_OV to see whether any look like copying heads, with mostly positive real parts. I also want to find the shared token subspace across layers, though that isn't strictly needed. The real test for shuffled versus as-is text is whether previous-token heads feed induction heads, since that composition would be useless once order is randomized.

I'm working out how to tell previous-token heads from local or broad ones in layer 0 using only the weights, since there's no positional embedding to read off directly. In the shuffled setting every position in a line is about equally informative, so a previous-token head would be no better than a broad one, and I'd expect the attention to spread over the current line. Telling those apart from the QK structure alone is the open question.

I'm weighing whether a trick can estimate the position subspace, since it is shared across all layer-0 heads, the layer-0 MLP, and later layers through the residual stream.

I'm now looking at whether the per-head QK matrices carry a shift-like structure over the position subspace, and I'm planning to compare their Frobenius norms to see which heads look most positional.

I'm working out whether the position subspace can be identified from the QK matrices alone, checking the symmetric and antisymmetric parts, the singular spectra, and how each layer's heads compose with earlier OV circuits.

I'm checking the row spaces of the layer-0 W_q and W_k matrices to see whether one head's directions look like position and another's look like token identity. If the heads cluster into those two groups, I can separate the subspaces. I'm also considering whether the layer-0 MLP and the OV circuits point the same way.

I'm estimating the position subspace P as whatever layer-0 heads read that isn't in the token subspace T, which the value-reading directions should reveal. Then I'd measure how much each head's QK energy falls in P×P versus T×T, though telling previous-token from broad heads still depends on knowing how the position embedding is structured.

I'm estimating how large the QK norm would need to be for a previous-token head to sharply peak on position i-1, since the LayerNorm-scaled inputs have norm around 20 and the score is divided by 8. Broad heads should show much smaller QK norms, so the norm could serve as a rough sharpness proxy.

I'm estimating the spectral norm of a random 64×384 query and key pair, and the bound comes out near 0.3, which is still well below what a sharp attention pattern would need. So for a random initialization the QK norm alone won't separate sharp heads from broad ones, and I'd have to project onto the actual embedding subspace first.

Working through the scale: a random head's scores come out around 0.15 in spread, so attention is close to uniform at init, and a trained previous-token head would need scores on the order of 5 or more to peak sharply.

I'm now looking at whether the QK matrix restricted to the position subspace P has a shifted-diagonal structure, since a previous-token head would need its scores to peak at the prior position. Without knowing the positional embeddings, I can't read that structure directly from the weights, so this stays an open question.

I'm weighing whether the positional structure can be read off the QK matrix restricted to the position subspace, with a learned position embedding that might be smooth or sinusoidal. A previous-token head would show an antisymmetric, shift-like component there, while a broad head would look more symmetric. This is getting deep, so I'll explore the data first to see whether clearer signals show up.

I'm counting how many characters the model actually uses: some letters like q, v, and x barely appear in romaji, while Python text leans heavily on spaces and newlines. Unused characters never get trained, so the effective token subspace the weights read should be roughly the number of characters that show up in the data, and I need to estimate that count from the weights alone.

I'm sorting through which romaji letters ever show up, noting that l, q, v, and x are nearly absent while c only appears before h. That leaves roughly 22 used letters, plus space and newline, so the effective token subspace should be about 24 dimensions. The open question is how to count that from the weights alone, since unused characters never get trained and their embedding directions stay near random.

I'm thinking about whether the token subspace can be pinned down from the weights, since layer-0 reads token and position together through LayerNorm. The trained directions probably span both subspaces, so separating them cleanly is hard. One possible handle is the copy structure: the OV circuits of copy-like heads should map token directions to the same tokens in the unembedding.

I'm checking whether the QK matrix's score for a query on one token and a key on the previous position stays large for matching tokens. That would make the composition of the induction-style QK with the previous-token OV map close to the identity on the token subspace, which would help isolate T.

I'm noting that if the induction pattern exists, the top positive-eigenvalue eigenvectors of the composed operator G would line up with the token subspace, and counting them could give the number of distinguishable characters. I'm also thinking about how the induction head's copy circuit should map each token direction back to itself in the output.

I'm going to start with the basics: per-head norms of the query, key, value and output weights, the spectra of each head's QK and OV matrices, and the symmetry of the QK structure. That should show which heads look sharp and which look broad before I go deeper.

I'm checking the layer-0 LayerNorm scale and bias on the residual dimensions, since large biases or small gains might mark out the notable ones. I'll also look at the MLP input weights and the projection that follows them, then write a first script covering the basic per-head statistics.

