Looking at the P3 directions, index 0 stands out as special with massive dimensions, likely a norm or confidence signal, while indices 1-23 look dense and uniform like typical output-token directions. That leaves exactly 23 dense directions, matching the 24 used tokens almost perfectly.

To be more rigorous I should strip out special dimensions (those with abnormally large residual values like 5, 21, 39, 53, 58, 90, 99, 184, 214, 219, 383) and recompute the SVD spectrum on the cleaned matrix to confirm the dense direction count holds. I also want to cross-reference this with other attention heads' OV circuits, like L3H4, which shows unusually large singular values, to see if a similar pattern emerges there.

