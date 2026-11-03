I'm estimating the token subspace at 20 dimensions and checking how each head's queries, keys, and values load on it. L0H5 has near-baseline Q and K in that subspace, so its attention runs on positional structure, while its V reads token identity strongly. That fits a previous-token head. L0H1 shows a similar pattern with a somewhat weaker V, so it may be doing the same thing, though I'm not yet sure what it attends to.

I'm checking the remaining heads: L0H4 is roughly uniform, while L0H0, L0H2, and L0H3 look partly content-based, since their keys read token identity to some degree. The layer-1 heads have strongly negative trace on the token subspace, which suggests they attend away from keys carrying the same token as the query, though I'm not yet sure what pattern that produces.

I'm looking at the layer-0 QK blocks, where the positional-positional term dominates for L0H1 and L0H5, while L0H0's key reads tokens to some degree. I'm also wondering whether the positional subspace could reveal how those prev-token-style heads attend.

I'm now asking whether the positional subspace could reveal the same previous-token pattern: if the position embeddings have a Fourier-like structure, the QK forms would act like rotations, and L0H1 attending two back would show a doubled-angle form compared with L0H5.

I'm checking whether L0H5's QK on the positional subspace looks rotation-like, with complex eigenvalues, rather than a plain bilinear form.

I'm weighing whether a shuffled corpus could still yield a prev-token head plus an induction circuit. Within a shuffled line the next character depends only on the leftover characters, which I can't see directly, so an induction pattern would have little to latch onto there.

I'm noting that the induction heads' key matching draws on several earlier heads, and that the circuit's token-identity structure points to sequential text rather than a shuffled corpus. Next I'm weighing which features of the circuit could tell the two kinds of corpus apart, starting with how the layer-3 heads match multi-character context.

I'm looking at the layer-1 heads next, starting with how their QK behaves in the non-token subspace, which is large for L1H1 through L1H5. Their K-composition with L0H0 and L0H4 is strong, and their output copies token identity into the induction heads' matching. I'm still unsure whether they're local heads, and why they would lean on those earlier heads rather than the positional embedding directly.

I'm now checking the Q-side singular vectors of M for L1H1, to see whether they lie in the token subspace, in L0H5's output space, or somewhere else.

I'm now reading L0H4 as a near-uniform averaging head whose value output mostly carries positional information, so its output is roughly a function of the current position. L0H0 attends to a particular token type with a recency bias, and its value copies the positional signal of that attended token, which could mark where the last such token sat. I'm wondering whether the layer-1 heads read these two outputs through their keys.

I'm thinking about whether the layer-1 heads could be picking out the current word, since their keys depend on where the last space or newline sat. That would be useful on any corpus, so it doesn't yet separate Python from English or romaji. I want to find a feature of the circuit that does.

I'm now checking the eigenvalue spectrum for L2H1 against L0H5: about 20 eigenvalues are clearly positive, then the values fall off into a noisy tail that goes negative down to about -5.5. The token subspace looks like roughly 20 to 22 significant dimensions, which fits a corpus with a somewhat smaller character set.

I'm checking the singular values of the value matrix for L0H5 restricted to the token subspace, and the full OV spectrum, to look for a knee that would show how many dimensions it really uses. I also want to see how the layer-0 MLP reads the previous-token output for bigram structure, since romaji's strong consonant-vowel alternation should show up there differently than in English or Python.

I'm considering which newline-related features could separate Python from English or romaji. Python's short lines and frequent indentation should make newline structure dominate, while long paragraphs would show it less. I'm wondering whether the indentation-copying behavior after a newline could be read directly from the weights, even though the newline token's embedding isn't visible to me.

I'm working through how a head might copy the indentation of the previous line after a newline, which induction only partly explains through multi-character prefix matching. I'm also considering whether newline and space structure separates Python from English, though space frequency alone doesn't seem to.

I'm moving on to a different angle: the local heads versus the induction heads, where romaji's syllable structure would call for exact attention to the previous one or two characters, a pattern English also uses at up to three back. I'm also considering whether the positional geometry of layer 0 could be isolated by taking the complement of the token subspace within the row space of the L0H1 and L0H5 query and key matrices, and then examining QK on that subspace.

I'm wondering whether the positional part of the layer-0 MLP input could be pulled apart from the token part, so I can check how it drives the neurons.

I'm checking how much of the first MLP's input weight energy falls on the token subspace versus the positional part, which would show whether it mainly handles token identity or position. I'm also weighing how much it reads the previous-token head's output, since romaji's rigid syllable structure would make bigram information especially predictive there.

I'm checking whether L2H2 is what separates the two corpora: its value barely reads the raw token identity, yet L3H5 and L3H1 match on its output with strong positive trace. I want to see how many dimensions that composition uses before I read anything into it.

I'm now weighing a head classification in which L0H5 is a previous-token head whose value copies token identity into the induction keys, and L0H1 is a local head with large positional QK and token-copying values. I'm still treating the L2H2 result as a hint that the composition reads richer context than raw token identity, so I'll hold off on drawing conclusions from it.

I'm assigning provisional roles to the rest of the layer-0 and layer-1 heads: L0H4 is a broad, near-uniform averager, and L0H0 leans on content while attending with a positional bias toward recent boundary tokens, with soft attention given its small QK singular values. L0H2 and L0H3 look content-driven but I can't pin their role yet, and the layer-1 heads may be attending locally or broadly, depending on how they compose with L0H0 and L0H4. On the layer-2 and layer-3 side, L2H1, L3H1 and L3H5 read as the copying circuit, while L2H2 is harder to place and may still belong to that circuit as an input to L3H5.

I'm now sorting the rest of the positional heads, L3H0, L3H2, L3H3 and L3H4, where the positional block dominates and little of the signal comes from the token. L3H3 looks like a broad averager, given its small QK norm and large OV. To tell previous-token heads from local or broad ones without the positional embeddings, I'm comparing each head's QK form on the positional subspace against L0H5's, since that's the known previous-token head.

I'm going to compare the QK matrices of all the heads on the positional subspace, using Frobenius cosine similarity, to see which heads share L0H5's previous-token pattern. Heads with high similarity would be candidates for the same role.

I'm checking whether L0H1's Q/K read the same positional frequencies as L0H5, and comparing their bilinear forms on the positional part. If the two share a block structure, I can tell whether L0H1 attends at a similar offset to the previous-token head.

I'm going to check whether the eigenphases of L0H1's form on the positional subspace are doubled relative to L0H5's, which would indicate an offset-2 head rather than a previous-token one. Since the positional embeddings aren't exactly Fourier and LayerNorm is nonlinear, this could come out too noisy to read.

I'm going to check whether the QK matrices of the different heads line up, using cosine similarity on the positional subspace.

