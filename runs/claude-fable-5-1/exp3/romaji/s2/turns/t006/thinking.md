I'm seeing very strong K-composition: every layer-1 head reads its keys from the outputs of L0H1, L0H2, L0H4 and L0H5, at roughly 6 to 12 times baseline. Those four layer-0 heads have similar OV circuits, so they seem to write a shared feature that layer 1 uses for keys. L0H3 and L0H0, by contrast, aren't read by layer-1 keys at all.

The value side looks different: L0H3's output is what layer-1 heads read for their values, so it carries content rather than positional structure. The layer-0 heads on the key side, H1, H2, H4 and H5, seem to build a shared feature, and H5's broad attention suggests it averages across the context. I'm guessing that feature tracks where the current token sits within a line, with layer-1 heads using it to attend within the line.

I'm now checking the layer-0 heads more closely: whether the QK directions of L0H1, H2, H4 and H5 share a key-side feature, and whether that feature lies in the same subspace the value and MLP paths read.

I'm going to compare the row spaces next: the top singular directions of L0H3's query and key weights, the MLP input weights, and the value weights, and check how much their principal angles overlap. If L0H3 reads a subspace the value weights don't touch, that would back up the idea that it's a positional head. I'm also considering a different route to the embedding subspaces, starting from the layer-0 LayerNorm output and noting that the later weight matrices only get gradient inside whatever subspace that output actually spans.

I'm working out that the trained part of W_fc.0 should live in a subspace of at most about 156 dimensions, since its updates are outer products with the layer-0 LayerNorm output. So its singular values should drop off sharply after roughly that many, with the rest being leftover random initialization.

I'm checking the full spectrum of W_fc.0 to count how many directions carry trained signal, expecting a sharp drop to the random bulk near 156 dimensions. I'd then compare the same count for the stacked query, key, value and MLP input weights to see whether they share a row space.

I'm now looking at whether L0H5's value weights, which give a few strong directions, read from the same subspace as the query and key weights of the other layer-0 heads. I want a common basis for the layer-0 readers, so I can then look for sub-structure among the directions many components share.

I'm now weighing how each corpus would shape the model's use of newline detection and position-in-line. Python's indentation and frequent newlines would push toward line-structure features, while English wiki windows rarely contain newlines, and romaji text leans on strong short-range character bigram statistics.

I'm looking at the L0H3 and L0H0 key ratios and checking whether any layer-3 head reads from L0H3 in its keys. L3H3 shows a K-composition ratio of 1.32 with L0H3, which is weak, and L3H4 is close to baseline, so there's no clear induction-style key reader here.

I'm checking whether layer-1 heads could be doing induction through a previous-token head in layer 1 instead of layer 0. Since only attention can move information between positions, I'm comparing K-composition of layer-2 and layer-3 heads against layer-1 OV circuits, given that L0H3's output already feeds every layer-1 head's values.

I'm now computing the full composition matrix between every pair of heads across layers, not just layer 0 against later layers. I also want to run the induction eigen-test on any pairs that show high K-composition, to see whether a previous-token head is hiding behind the shared L0H1/H2/H4/H5 feature.

I'm considering whether a rank-1 query-key matrix with a positional ramp would produce recency attention, but causal masking means the current position always competes, so this alone would favor the current token rather than the previous one. To pick out the previous token specifically, the current position would have to be suppressed somehow.

I'm checking whether a recency-style head would need to exclude its own position to act as a previous-token head. I'm now looking at the structure of L0H1, H2 and H4's QK matrices to see whether the top singular vectors on the query and key sides match (a symmetric u uᵀ form) or differ (a u vᵀ form). The earlier numbers, with a symmetric part of 5.87 against an asymmetric part of 4.82, suggest the symmetric piece isn't clearly dominant.

I'm wondering whether the token and position subspaces can be told apart without labels. The MLP0 neurons each read one direction, so if many of them cluster around nearly the same direction, that cluster might be a character detector, and its center could expose the token subspace.

I'm checking whether L0H3's OV spectrum has a clear cutoff, which would show how many directions it actually uses. The flat values around 4 suggest high rank, so I'd print the full spectrum and count the directions above the noise floor.

I'm going to print the full singular spectra for the layer-0 heads' QK and OV matrices, along with W_fc.0 and the stacked layer-0 reader matrix, to see how many directions carry real signal.

