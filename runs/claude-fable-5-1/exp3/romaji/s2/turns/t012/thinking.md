The trained MLP0 rows' spectrum decays smoothly over about 50 dimensions, with no clear knee at 27. The stacked W_v.0 H3+H0 has roughly 23 singular values above 1.3, then a drop to 1.12 and 0.98, which points to about 23 to 24 dimensions. That could be the effective token subspace, but with a 28-token vocabulary and mean removal the maximum would be 27, so I'm wondering whether a few rare tokens are pulling the count down.

I'm checking whether the romaji idea holds up: if l, q, v, and x are unused, that would leave about 23 tokens, matching the singular value count. It's suggestive but weak, since rare letters in English or Python could lower the rank too. I'm also considering whether MLP0 neurons align with individual token directions, so I'll compare the cosine similarities among the trained fc.0 rows and look for tight clusters.

I'm comparing the singular value spectra of the MLP0 and MLP1 weight matrices, along with their output norms, to see whether layer 0 carries the heavy bigram-style load I'd expect in an as-is model.

I'm checking whether the layer-1 heads are near-duplicates in their statistics, since that would suggest a simple, uniform task rather than the specialized heads I'd expect in a natural-language model. I'm also weighing the layer-0 weight growth against the layer-1 numbers to see which layer carries more of the learned signal.

Layer 2 looks uniform as well, and layer 3 splits: H0, H1, H2, and H5 resemble each other, while H3 and H4 differ, with H4 acting as a broad head because of its small QK norm and large OV output. Taken together, layer 0 seems to have a previous-token or local head, a broad head, three delimiter-position heads, and one head whose role is still unclear.

I'm weighing the uniformity of the layer-1 heads against the idea of a simple counting task over a positional window, which would fit shuffled text. A natural-language or as-is Python model would likely develop varied heads, including induction, and there is none here, so I lean toward the shuffled-text explanation. Next I want to settle which corpus it was.

I'm now asking what separates the candidate corpora under the shuffled-text reading. Python lines are short, with median length around 30 to 40 characters including indentation. English wiki text has long paragraphs mixed with short headings and list items. Aozora romaji paragraphs vary widely, and romanized Japanese runs about 2.5 times longer per paragraph than the original characters.

I'm weighing whether line length can tell the candidate corpora apart: Python lines are short, English paragraphs run long, and romanized Japanese paragraphs fall somewhere between. Space fraction might help too, since indentation and punctuation-turned-spaces push Python's share above English's, while romaji word lengths depend on how the text is segmented.

I'm wondering whether the L0H1, L0H2 and L0H4 heads might actually key on something other than newlines, since shuffled text makes word boundaries meaningless. Their key directions have some overlap with the token subspace, and L0H5's key sits strongly in it, so the positional-copying reading may not hold.

I'm wondering whether these heads key on newlines and copy position, so that layer 1 could recover each token's offset within its line by combining the line-start position with the absolute position.

I'm weighing whether the layer-1 heads could be doing a windowed count over the current line, where the model only sees a line start position and an absolute position, so attention mostly spreads across the whole window. For English text with long lines, those line-start cues would rarely appear, so the learned pattern would fall back to a broad, unigram-like mix.

I'm weighing whether romaji line lengths of around 60 to 300 characters would still hit a newline inside most 128-character windows. Python code, with comments and docstrings stripped, has much shorter lines, closer to 30 characters, so its newlines would show up even more often.

I'm trying to tell python apart from romaji by their character inventories, and I'd need the token embedding to read that directly. The count of effective token directions is the handle I have: the stacked value readers show about 23 to 24 singular values above the floor. Romaji with a space and newline would use about 24 symbols, so the count fits, though LN mean subtraction could shift it by one.

I'm weighing whether the token directions really carry the rarer letters, since q, x, z, and j would have low variance in English and so contribute little to the spectrum. That means a count near 23 or 24 is weak evidence and could come from almost any corpus. A sharper test would be to ask what MLP0 computes when shuffling removes useful bigrams: the next-character distribution should track how character counts co-occur within lines.

I'm estimating the token subspace dimension from the L0H3 value singular values, looking for where the spectrum drops off toward the noise floor.

I'm counting how many singular values of the L0H3 value read sit above the noise floor, which comes out to roughly 20 strong directions plus a handful of weaker ones. The weaker ones could be position rather than token, so the count isn't clean enough to pin down the character inventory.

The layer-0 LN weights amplify the position dimensions and damp the token dimensions, so attention at this layer leans on position more than on token identity. That fits a shuffled-text setup, where the current token says little about what comes next, though an as-is model's bigram path through MLP0 would normally need the token signal.

I'm checking whether I've got the roles of the two read subspaces backwards: if Q3 is the token subspace rather than position, then L0H5's value read would be a bag-of-characters composition, which is what a shuffled-text model would want.

I'm working out which of L0H3's two read subspaces is the token one: its query and key both read the Q3 directions, which points to content-based matching, while its value reads T3. I'm considering whether the head copies positions, but T3 is nearly orthogonal to Q3, so that seems less likely, and I'm checking which subspace the MLP's reading supports.

I'm settling on the token subspace: MLP0 has to read the current token, since position alone wouldn't give it anything to work with. The T3 directions also number about 20, close to the 28-symbol vocabulary, so I'm favoring T3 over Q3 for token identity.

I'm comparing whether the L0H1, L0H2 and L0H4 heads copy token identity or position: their keys sit partly in the token subspace, which fits finding a delimiter and copying its position, but a rank-1 position-to-position pattern could instead be a recency head that copies nearby characters.

I'm checking whether T3 or Q3 is the token subspace by looking at the layer-0 LayerNorm gains: the token directions carry small gains and the position directions large ones, which could mean the gain rebalances whatever raw norm the token embeddings have.

I'm weighing T3 against Q3 for the token subspace: T3 shows a clearer drop around 17 to 20 directions, which fits a 28-symbol vocabulary, while Q3 keeps going past 30. Layer-1 reads of the residual stream would be a further check on which subspace carries token identity.

I'm checking the shuffled-model hypothesis against the weights: if the model is estimating per-character counts within the line, the later layers should read token identity only weakly, which is roughly what the L2 and L3 reads and the MLP1 numbers show. Before trusting that, I want to pin down what the ideal algorithm predicts for the weight structure.

I'm now checking whether L0H3 should be read as a local window head, averaging over the last ten to twenty positions, which would fit its flat, high-rank spectrum.

I'm asking whether L0H3's value read is a sharp shift-by-one pattern, since its flat spectrum points that way rather than to a smooth boxcar. In a shuffled-text model previous-token identity shouldn't matter much, except possibly to mark where a line starts after a newline.

I'm now asking whether L0H3 might just be an ordinary previous-token head, which would mean the model is not shuffled after all. In a standard model, copying heads should have positive value-output alignment with the token subspace, so I'm checking whether these heads' suppression pattern holds up against the vocabulary alignment of W_E and W_U.

I'm considering whether induction could run with the roles reversed: the query reads the previous token from L0H3, and the key reads the current token from MLP0, so a head would attend to a position where the token matches what came before the current one. The K-composition with MLP0 is strong enough to make that plausible, so I need to check whether the query-side reading actually holds.

I'm looking at which L1 heads attend by position versus token: their keys lean on the position features from L0H1, L0H2, L0H4 and L0H5 rather than the previous-token head, so attention is driven by where tokens sit, not what they are.

I'm now testing an induction variant with the roles swapped, where the query reads the previous token, but that version would predict the token before the match rather than the one after, so it's useless for next-character prediction. The L1 keys are driven by position features rather than by the previous-token head, so the attention is positional, not content-based induction.

I'm ruling out induction for the as-is case: a 4-layer model trained on this much text would almost certainly form induction heads, and the L1 keys are driven by position features rather than the previous-token head. I'm checking whether K-composition routed through the L1 outputs could carry that information instead, and the L2 numbers say it doesn't.

I'm now seeing that the "all later heads negative definite" pattern doesn't settle the sign question, since whether W_E and W_U align depends on whether repeating a token makes it less likely to appear again in a shuffled line. So the sign of the OV eigenvalues can't be taken as evidence on its own, and I'm treating the no-induction argument as the stronger point.

I'm now weighing which corpus this would fit under shuffling. Python lines are short and almost every 128-character window contains a newline, while English Wikipedia paragraphs run much longer, so line-boundary features would matter far less there.

I'm weighing how many token directions the residual subspace really uses, and I'm thinking about the corpus this might be. Romaji lacks l, q, v and x, which would leave about 24 effective tokens against 28 for Python and English, so the vocabulary alone could help tell them apart. I'm planning to check the layer-0 readers, the MLP and value weights, to estimate the subspace dimension directly.

I'm computing the singular spectrum of the composed matrix from the L0H3 value output through the L1 value and MLP readers, to see how many token directions actually pass through. I'll also estimate the token subspace dimension directly from L0H3's OV singular values, which are fairly flat across the first six.

I'm weighing how many tokens the value subspace actually carries: the L0H3 spectrum has about 17 strong directions plus a few medium ones, which suggests roughly 20 distinguishable tokens. Romaji would give about 22 to 24 with spaces and newlines, while English would use a different set of common letters, so the count could help separate the candidate corpora.

I'm comparing the vocabulary of each candidate corpus, since rarer letters like j, x, q and z would drop out of the token count. Then I'm checking the tail of the L0H3 singular spectrum, where values near one need to be weighed against the initialization noise level.

I'm reading the L0H5 output as a rank-two signal that carries the query-side position feature, so it looks like an absolute position code that later heads use as keys. That fits a relative-position scheme built from keys over positions.

I'm checking whether the L0H1, L0H2 and L0H4 key directions point at the same token: if they all lock onto newline, that fits a shuffled corpus where only line boundaries matter, and if they split between space and newline, the picture is different.

I'm checking whether the L0H1, L0H2 and L0H4 key directions all lock onto the same token, and comparing them against L0H5's weak key preference and the directions MLP0 reads most strongly.

