I'm checking the value magnitudes for the W_v.0 heads with Q3 projected out: about two dozen are at or above 1.5, and the rest taper down to around 0.7. That suggests roughly 24 to 28 trained dimensions in the complement that the value projections read. Some of those are likely leftover position dimensions, since Q3 only covered 40 of the up to 128 position dimensions.

Next I'm weighing the L1–L3 heads: whether their keys are dominated by L0H5's broad positional average or by the sharper delimiter-position signal from L0H1, H2, and H4. I plan to compare the key-side energy fractions in each subspace to settle that. Then I'll finalize the head labels, marking L0H0 as unclear for now.

I'm assigning labels to the remaining heads: the L0 content-keyed heads (H1, H2, H4) look broad, since they spread across all earlier occurrences of the delimiter token. L0H3 is previous-token, L0H5 is near-uniform so broad, and the L1 through L3 heads mostly read as broad too. L3H3 is still unclear because its key signal is tiny, and I'm weighing whether the L1 and L2 heads should split into a few different categories rather than all sharing one.

I'm now weighing whether the L1 and L2 heads should be labeled 'local' rather than broad. Their keys carry L0H5's smooth position average, which is monotone in position and could produce a recency bias, so I want to compare how much of each head's key energy comes from L0H5 versus the delimiter-position heads L0H1, H2, and H4.

