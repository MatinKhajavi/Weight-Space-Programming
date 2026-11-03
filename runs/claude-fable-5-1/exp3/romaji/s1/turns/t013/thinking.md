I'm checking which layers actually read the token subspace: layer 1's Q/K/V are well above chance, while layers 2 and 3 are close to chance, so they probably rely on features the MLPs produce in other directions. Next I want sharper evidence, starting with whether the behavior looks like a count-based, bag-of-characters model that tracks line length.

I'm thinking about whether the later heads' queries can match on line membership. They don't read the layer-0 position dimensions, so they seem unable to pick out "same line" tokens directly, though a recency preference could still produce a similar pattern. That makes this check inconclusive, so I'll try the L0 key direction next.

I'm now weighing whether MLP0's neurons mix the current token with the previous-token signal, which would suggest n-gram modeling. But the previous token should carry no special weight in shuffled text, since every character in the line is an equally informative sample from the same multiset.

I'm now leaning toward L0H3 being a shift head that attends to the previous token rather than to itself, since the antisymmetric part of its matrix is large and attending to self would be redundant with what the MLP already reads. MLP0 depending on it at about five times chance, plus layer-1 queries and keys both picking up the previous token, looks like the pattern of induction-style circuits, which points toward "as is" over shuffled text.

I'm trying to pin down which signal is discriminating: the previous-token head L0H3 feeding MLP0, plus the layer-1 heads reading the previous token on both query and key sides, looks more like induction or n-gram matching than a bag-of-characters model. A separate check would be whether the romanized Japanese corpus's smaller letter set (no l, q, v, x) shows up as a smaller effective token dimension, though measuring that needs the embeddings.

I'm checking whether the 3 near-duplicate heads L0H1/2/4 locate newlines and their positions, since line structure looks central to the model. Python's indentation, with leading spaces after each newline, would make that structure matter a lot, which could point toward Python rather than English prose.

I'm weighing whether the heads at L0H1/2/4 select newline tokens or spaces, since both would be word-or-line boundaries that show up across the corpora I'm comparing. Telling them apart from their key direction alone is hard, because either choice would live in the same token subspace.

Still thinking about whether the newline-selecting heads matter: knowing where the last newline sits gives the column, which would matter for Python indentation, and the mean position of all newlines is less useful than the most recent one unless a second, recency-driven component adds weight to the nearest match.

I'm checking whether the second singular vectors of H1's QK land in the positional subspace, and whether k1's small positional share at k=20 is meaningful or just noise. I also want to see whether these heads are picking out spaces rather than newlines, since average space position would give a different signal. The output directions of L0H1 and L0H5 look correlated across the same dimensions, with H4 and H5 closely aligned, and dimension 21 seems to be the main channel the later heads read from H5.

I'm weighing the romaji idea next: its strong consonant-vowel alternation makes the previous character very predictive, which would favor a previous-token head like L0H3 over a bag-of-characters model. Python's long-range repetition and indentation structure would instead lean on induction and newline-position heads, so the corpus question depends on which of those dominates.

I'm weighing whether the Python hypothesis holds up: its frequent newlines and indentation would make the line-start heads central, though I can't yet confirm which tokens they select. I'm also estimating the token-subspace dimensionality, since romaji leaves out several letters, so their embeddings would only be shaped by weight decay.

Now I'm weighing the space token as a signal: spaces make up a large share of characters in English, romaji, and Python alike, so their frequency alone won't tell the corpora apart.

I'm considering whether layer-1 heads could implement an order-2 induction match: MLP0 combines the current token with the previous-token signal from L0H3, and the layer-1 Q/K read that combined feature, which would let a head match earlier contexts. The composition result showing Q and K both reading the token-identity subspace at 2-3x chance fits a same-character matching head, so this is worth testing directly.

I'm testing for induction-like structure in layer 1 by looking at whether the query reads the current token's identity while the key reads the previous-token signal that L0H3 writes. If that holds, the bilinear form between those subspaces should be close to identity-like along the token dimensions.

I'm planning to compute, for each later head, a trace-to-norm ratio on the bilinear form restricted to the token subspace, to see whether it's identity-like, and to check the eigenvalues of its symmetric part for positive-definiteness. The comparison is against a same-token version of the same matrix.

