Switching to the orthonormalized prev-image (Aq) changes the picture. For L3H1, token-to-token matching is 0.38, but token-versus-previous-token is slightly negative at -0.05, while previous-to-previous is 0.41 and head-3 matching is 0.62. So this head looks like a content-matching head: it attends to the same token, the same previous token, and the same head-3 state together.

I'm checking whether this head keys on the matching position itself or on the one after it, since a causal attention at j can't see tok_{j+1} in the residual there. The earlier non-orthonormalized test gave 0.46 for the token-versus-previous-token match, but the orthonormalized version drops it to -0.05, which I suspect comes from how the QR basis weights the singular structure of OV_H2.

I'm now checking which of the tests are actually valid: the orthonormalized basis only lines up across both sides when the same basis is used, so the cross-token comparison from Aq is meaningless, while the same-basis prev-to-prev identity test still holds.

I'm now testing whether L3H1 matches the same token and also the token before the matching position, which would explain how it picks up the next token. The non-orthonormalized check gave a positive signal across all 30 diagonal entries, so I'm reconsidering whether that version is the one to trust.

I'm now looking at whether L3H1's matching could be shifted, since the key at j would need tok_{j+1} to predict the next token, and that isn't in the residual at j. I'm also asking whether L0H2 attends to both i−1 and i, which would explain the mix of same-token and shifted-token signals.

I'm weighing whether L3H1 could be an induction-style head, where the key at j encodes the token before it, so that tok_i matching tok_{j-1} would point to tok_j as the prediction. That fits the 0.46 signal for tok_i against prev_j, but the 0.41 for prev_i against prev_j doesn't fit cleanly, since proper induction would need tok_{i-1} to match tok_{j-2}, which the residual doesn't expose linearly. So the two matches seem to pull in different directions for the same j.

Now I'm weighing whether L3H1 uses a soft mix of same-context and shifted-context matches, since the same key position j shows both tok_i matching tok_j and tok_i matching prev_j. L0H2 might also split its attention between i−1 and i, which would blur those signals together.

Layer-1 heads show negative same-token and same-previous-token scores, which suggests they avoid positions with the same bigram context as the current one, the reverse of copying, or possibly just avoiding self-attention.

The anti-self pattern shows up in all six heads at about −0.4, so a self-exclusion explanation looks more plausible than a context-matching one for layer 1. Including the current token would just dilute a local average, but a content-based anti-self term also works at any position without relying on positional cues.

I'm settling on the layer-1 heads: the anti-self pattern of roughly −0.4 across all six makes self-exclusion the more plausible reading than context matching. Layer 2 shows only slight anti-context, and layer 3 heads 0 and 1 are the ones doing positive context matching. MLP0 looks like it re-encodes the current token in its own directions.

I'm now weighing which corpus would best produce these patterns: one strong induction-style context matcher in layer 3, with the local machinery in layers 0 and 1 handling the rest. Python seems likely to build induction heads earlier and more strongly, since repeated identifiers are so common, whereas natural language would leave this kind of matching less developed.

I'm weighing whether the romaji and English corpora would both produce the induction-in-last-layer pattern, with particles handled by n-gram-style machinery. Python seems less likely, since its induction would probably form earlier and more strongly.

I'm considering whether MLP0's underuse, with about 1025 neurons near initialization, reflects how sparse the character bigram table is in romaji, where few consonant-vowel combinations are allowed. English has a much richer bigram structure, and Python identifiers fall somewhere in between.

I'm wondering whether MLP0 would encode bigrams better on romaji, though it only reads H2 weakly anyway. The layer-0 MLP seems to have a light job, mostly re-encoding the current token with some bigram features. Romaji's consonant-vowel transitions are nearly deterministic, so those are simple to learn, while English bigram statistics are richer but still manageable with a few hundred neurons.

I'm weighing whether a head that attends to word boundaries, like spaces, would show up for English or romaji, since both have clear word breaks, while Python's newlines and indentation would push its machinery elsewhere. Based on the absence of position-copying in layers 1 through 3, I'd still put Python at around 20%, and I'm checking whether a rank-1 key-side detector for spaces appears in any head.

I'm checking whether the dominant key directions of the layer-1 heads all point the same way, which would mean they're detecting one special token like a space or newline. I'd compare the top singular vectors of each head's key projection pairwise, and also see how that shared direction lines up with MLP0's principal direction.

I'm checking whether the strong MLP0 direction, with its large singular value of about 17.8, lines up with a space or newline token, since that would be the most common and most salient token to encode.

