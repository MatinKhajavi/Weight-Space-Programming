Now I'm noticing that every head in layers 1-3 has an OV matrix that's almost entirely negative semi-definite, with the negative parts far outweighing the positive ones. That means each head is effectively subtracting a weighted sum of the attended, normalized inputs rather than copying them. I'm starting to check whether the layer-3 heads follow the same pattern.

I'm checking whether the layer-3 heads that erase the L0 outputs are actually cancelling those specific attention-head contributions. If a head writes a negative copy of an attended position's normalized input, the downstream MLP could read the difference from the residual, which would act like a same-token check. I'm also wondering whether this subtraction is really suppressing a predicted character that has already appeared in the line.

I'm now thinking that if the remaining letters are just the line's multiset minus those already seen, then each character's probability should drop with every occurrence, so the optimal predictor would subtract seen counts from the expected composition. A broadly attending head that subtracts the attended token identities would implement that exactly, which fits the all-negative OV pattern in the later heads. I'm also wondering whether ordinary text models with this same negative pattern would look different, since copy-style heads are usually positive there.

I'm checking whether the embedding and unembedding directions are aligned, since that determines whether a negative OV eigenvalue reads as anti-copying. Without a reason for them to be anti-aligned, the sign flip would need an explicit explanation, so I'm weighing whether untied models tend to produce that.

I'm thinking that a negative OV eigenvalue on the token subspace would mean each head writes the negation of an attended token's embedding. That would make sense if the direct path already pushes against repeating the current token, since doubled characters are rare in most text, so the model would learn a negative alignment between the embedding and the unembedding for that character.

I'm realizing the sign of an OV eigenvalue in residual space doesn't settle whether a head copies or anti-copies, since that depends on how the embedding lines up with the unembedding. So I need another reference point for the sign, and the layer-0 MLP's next-character predictions are a candidate.

I'm noticing that the sign of an OV eigenvalue in residual space can't tell me on its own whether a head copies or anti-copies, since that depends on how the embedding lines up with the unembedding. The fact that all eighteen later heads share the same negative-definite pattern suggests they follow one sign convention, so I'm checking whether a shared alignment like W_out ≈ -W_E could explain it.

I'm weighing whether the sign of the OV eigenvalues can even tell copying from anti-copying, since that hinges on how the token embedding lines up with the unembedding. If the direct path already disfavors repeating a character, a negative alignment would flip the sign, so the uniform negativity across heads could still reflect a single consistent role.

Now I'm weighing how the spaces and doubled letters in English and Python would shift the sign of that direct-path alignment.

I'm weighing whether untied models tend to show a negative diagonal in W_E W_U, since a token rarely following itself would push the direct path against repetition. Even if that holds, the sign convention for the OV eigenvalues stays ambiguous until I can pin down that alignment.

I'm checking whether the MLP0 bigram path could serve as a sign reference, since its outputs are read through the unembedding in the same way the next-token predictions are. I'm also reconsidering the induction angle, because I haven't yet ruled out a previous-token composition with L0H3 under another setup.

I'm checking whether any layer 1-3 head has K-composition with the token-copying layer-0 heads, since that's what a classic induction circuit would need. None does; the only K-composed sources copy position features instead, so this model seems to lack induction entirely, which would fit a shuffled setup or a model that never learned it.

Next I'm checking whether the layer-1 queries read the positional subspace or the token subspace, since that decides whether their attention is positional or content-dependent. I also want to see how much of the value input comes from the L0H3 previous-token signal versus the current token's direct embedding.

Now I'm checking whether the layer-1 heads' attention is driven by newline-token keys from L0H1/2/4, and whether the value side mixes the previous-token and current-token encodings in a way that would aggregate identities across the line.

I'm wondering whether L1–L3 heads attend within the current line, so I'm asking whether corpus differences would show up in how lines are segmented. Python lines are short and indented, English lines run far past 128 characters with newlines rare in a window, and romaji lines fall somewhere in between, with a mix of long paragraphs and short dialogue lines.

I'm working out whether the line-length differences across English, Python, and romaji would change which heads need newline detection at all. With long English lines, a window is mostly a bag of characters, so the model would just average token identities. Yet three layer-0 heads are devoted to position-copying and later heads all key on them, which points toward line-structure sensitivity.

Checking whether the shuffled Python lines, which keep their characters and length but lose order, would need the model to track line boundaries and count characters already seen, since sampling without replacement inside a short line is a strong effect and newline prediction depends on how far into the line it is.

I'm checking whether the layer-0 heads' newline-position copying and the layer-1 to layer-3 heads' within-line attention could account for the shuffled-line structure. I'm also asking whether a broad head over recent positions, rather than a true previous-token head, could explain the L0H3 pattern.

I'm checking the L0H3 pattern against a shuffled-line hypothesis: if it's a boxcar-style local head summing recent characters, its output would double-count with the L1 value reads. I want something sharper to decide between that and a true previous-token head, and I'm leaning toward MLP1's reliance on the L0H3 output, which sits at roughly 1.55 in the ratio.

I'm wondering why the MLPs would avoid reading the L0H1/2/4/5 positional directions, since a ratio below 1 means less than chance. If those features are only consumed through the keys of later attention heads, then the MLP avoidance is itself informative about what the model is using them for.

I'm asking whether the MLP0 bigram-like behavior would still show up under shuffling, since in as-is text the current character strongly predicts the next one while in shuffled lines it mostly just reflects the line's overall character composition.

I'm now considering whether MLP0 could still carry bigram-like signal under shuffling, or whether the attention heads would do nearly all the work there. I can't read that off directly from the neuron counts or the W_fc spectrum alone. I'm also wondering whether MLP0 in the as-is case takes the current token and writes output aligned with the next-token unembedding, though without W_U I can't check that yet.

I'm weighing whether L0H3's QK acts as a sharp shift operator (a true previous-token head) or a smoother local kernel, since the spectrum of that position map should separate the two. Whichever it is, it feeds MLP1, which reads it most strongly, so that dependence is what I need to pin down next.

I'm checking whether L0H3's QK spectrum looks like a shift operator: the nearly flat singular values over the first fifteen or so dimensions would fit a unitary-like rotation, which is what a true previous-token head would produce, rather than a smoother local window.

I'm checking L0H3's QK spectrum for a shift-by-one structure, but the mix of symmetric and antisymmetric parts (22.7 vs 25.9) doesn't settle it yet. A pure "attend to self" or smooth local kernel would look different, so I'm weighing which one the numbers fit.

Checking whether L0H3's QK is a true shift-by-one operator is hard without the actual position vectors, so the spectrum can't settle it alone. I'll look for a different angle to tell the shuffled-line case from the as-is case.

I'm considering whether the L1 heads could be keying on the position of each line's start, so that attention favors the most recent newline rather than strictly the same line. The alternative would be a smooth average over positions acting as the key, which would give a different signature.

I'm checking whether the L1 queries' sensitivity to the L0H3 output, at about twice baseline, means they carry previous-token identity alongside the current token, and whether that could feed the keys in a way that separates the shuffled-line case from the as-is case.

I'm checking how much of the query and key signal comes from the direct token and position paths, not just the attention heads. To do that I need to undo the layer-norm scaling, so I'm mapping the L0H3 value directions back into the normalized residual space and then re-orthonormalizing them.

I'm projecting the layer-1 query weights onto the token subspace mapped back into residual space, but first I want to check whether dividing by the layer-0 LayerNorm weights amplifies noise, since those weights are quite uneven across dimensions.

I'm projecting the residual-space token subspace through diag(1/w0), though those layer-0 LayerNorm weights vary a lot and the inversion could amplify noise. I'll also try a cruder version that uses diag(w0) directly and compare how much of the query and key energy each subspace accounts for across heads.

I'm checking how much of each later head's query, key, and value weights falls inside the token and position subspaces mapped back into residual space, comparing against the random baselines of about 5% and 8%.

