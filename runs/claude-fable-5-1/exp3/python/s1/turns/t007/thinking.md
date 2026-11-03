I'm comparing the query-key matrices across layer 0's heads, and most pairs come out nearly orthogonal. The exceptions are L0H0 with L0H3 and with L0H2, which show moderate overlap. I'm also checking L0H5 against its own transpose, where the value is negative, suggesting a somewhat antisymmetric structure.

The next part is a worry about whether the L0H1 to L0H5 transpose link actually fits a forward-looking pattern, since the causal mask means an offset of -1 can't be reached. I'm thinking the near-zero cosines between layer-0 heads may just mean each head covers a different relative offset, even when they draw on the same positional subspace.

I'm now asking what the 2D subspace behind the L1 heads' pattern actually is, starting with L1H1: check the top singular directions of its QK composed with L0H4's OV, and whether they sit in the positional part or in the token outputs. I'm also weighing an alternative, that these heads just attend to the start of the current word or the previous line, and noting that L0H4's top positional value directions look like they're reading smooth position signals.

I'm reasoning that the uniform attention of L0H4 yields an output that is a smooth function of absolute position, decaying from early positions. L1's keys can then read that position-dependent signal while its queries read position directly, so the attention is positional but built through composition. I'm wondering why the model would do this rather than use the position embedding directly, and suspect the uniform head's output carries position more cleanly or with an amplitude that varies with position.

I'm wondering whether the L1 heads are positional heads that attend to a specific offset, with L0 outputs helping set that up. Their OV circuit would then carry the token identity from the attended position forward, which is what feeds the induction matching.

The L1 heads look like local heads reading different offsets and passing token identity to the induction keys. L0H0's positional output could give each position a sense of distance from the last word boundary, so the L1 heads may be matching on word-relative positions.

Now I'm checking the spectrum of the token-side subspace for the previous-token head, where the singular values stay fairly flat across its dimensions. Its value vectors read from both token and positional directions, though they lean toward the token side more than a uniform baseline would.

I'm checking whether the 22 token-side value directions of L0H5 really separate from the positional noise, since the unused residual dimensions could be inflating the apparent spread. Working through the weight-norm arithmetic, the random init noise accounts for a meaningful share of L0H5's value norm, so the clean token reading is weaker than the raw ratio suggests. I still need a test that can tell the corpora apart before I can say which one this is.

I'm weighing which observable signals could tell the three corpora apart: romaji would need offsets i-1 and i-2 for digraphs like sh and ch, English more like i-1 through i-4 for common 4-grams, and Python a similar range plus line structure. I'm also comparing how strong the induction mechanism is relative to the local one, since Python should lean heavily on copying identifiers, English less so, and Romaji somewhere in between.

I'm checking whether L0H4's uniform averaging carries something corpus-specific, like a running summary of token types, instead of just position. The positional values look stronger than the token ones, so I'm not sure the average says much about the text itself.

I'm thinking about whether the token-side structure of the MLP input weights shows clusters, since that could separate vowels from consonants or spaces from letters. If the W_fc.0 rows concentrate along one or two directions within the token subspace, that would hint at which corpus this is.

I'm now weighing whether the token-subspace eigenvalues of the combined matching matrix can say anything about which corpus this is, since those eigenvalues don't map cleanly onto individual tokens. I'm also considering the vowel/consonant axis as a possible corpus signal, but I can't yet tell which direction in the 22-dimensional token subspace corresponds to which feature.

I'm considering whether the Hepburn romanization's missing letters could be a corpus signal, since 'l', 'q', 'v', and 'x' would be nearly absent and their embeddings barely trained. I'd want to check how the W_out rows for those tokens look compared with the common letters.

I'm working out whether the 22 Hepburn letters plus space and newline, about 24 tokens, could be picked out from the weights, since letters like l, q, v and x would barely be trained. Counting the candidate alphabet, I get 22 letters plus space and newline, so the token set itself is small enough to be a distinctive clue.

I'm weighing whether the number of distinct tokens could separate the corpora: Python would have about 28 with x, j, q and z appearing fairly often, while romaji would have about 24 with those letters nearly absent. The induction spectrum shows roughly 20 clear eigenvalues, which is too few to tell 24 from 28 cleanly. I'm also turning over whether line length matters, since Python's short lines would make the positional embedding need newline detection, unlike longer English or romaji lines.

Next I'm thinking about line length as a corpus signal: English Wikipedia paragraphs run several hundred characters per line, and the romaji Aozora text likely runs 100 to 200. Python lines average about 35 characters, so a 128-character window would hold several newlines and the model would need machinery for line structure and position-in-line.

Now I'm trying to tell whether L0H0's key token is a frequent one like space or a rarer one like newline, since a head that reports the position of its last match needs sharp attention whichever token it targets.

I'm weighing whether the MLP input directions within the token subspace show a dominant class axis, like whitespace versus letters or vowels versus consonants, since that would mark which corpus this is. I'm also asking whether the induction heads' queries pick up word-position features when deciding what to copy.

I'm checking whether the spread of W_fc.0 rows across the 22 token directions has one dominant axis, which would point to a special token like space or newline that many neurons respond to. Identifying which token that is would be the hard part, so I'm considering whether the near-orthogonality of the embeddings could help match the matching-matrix structure to specific tokens.

I'm thinking about treating the matching matrix on the token subspace as roughly a sum of token directions weighted by how useful induction is for each token. If the embeddings are near-orthogonal, its eigenvectors would line up with individual tokens, which could let me map the QK structure of the L1 heads onto specific token-to-token attention patterns.

I'm checking whether the 22x22 content QK matrix on the token subspace has a low-rank block structure, like vowels against consonants, by looking at its singular values in an arbitrary orthonormal basis. A dominant rank-1 component would suggest a strong class split, which I'd expect to be clearest for romaji and weaker for English. I'm also doubting the near-orthogonality assumption behind mapping eigenvectors to individual tokens.

I'm checking the 22x22 content block on the token subspace of the L1 heads, and its trace-to-norm ratio sits near -3 against a -4.7 benchmark for a pure negative identity. That would mean these heads mostly avoid attending to positions carrying the same character as the current token, which is unexpected enough that I'm wondering whether my assumption about the subspace is off.

I'm checking whether the L1 heads' negative self-attention on the token subspace is an artifact of how T was built, or a real anti-duplicate pattern. It isn't obvious what a head that avoids same-character positions would be useful for, so I'm considering local n-gram roles as one possible explanation.

I'm wondering whether the L1 heads' strong avoidance of same-token positions, with trace ratio near -3, is a real anti-duplicate pattern or just a quirk of how the token subspace was built. If it's real, doubled letters in romaji (sokuon like "kk" or "tt") and English double letters would be the natural cases where it matters, and I'm also considering whether these heads might be locating offsets through positional composition instead.

I'm breaking the L1 heads down further, checking whether their queries and keys draw on the token subspace, the outputs of the layer-0 heads, or the MLP0 output directions.

I'm building a positional subspace P0 from the complement of T, using the Q and K rows of L0H1 and L0H5, keeping about 30 top directions. Then I'll measure how much of each layer 1 to 3 head's Q, K, and V energy falls into T, P0, the MLP0 output, and each layer 0 head's OV output.

