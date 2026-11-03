I'm seeing strong K-composition feeding from L0H1, H2, H4, and H5 into most L1 and L2 heads, plus some L3 heads, while L0H3's purely positional output seems to bypass this composition, feeding MLPs instead. L0H5 stands out since it attends almost uniformly and essentially outputs the context average, which downstream heads then use as keys.

My hypothesis is that L0H1, H2, H4, and H5 all write into a shared low-rank subspace, possibly encoding something like distance to the last newline or positional structure, that later heads read to decide things like "same line" attention patterns. I want to check this by comparing the SVD spectra of each head's OV matrix to see how much their top output directions overlap, and then see which of those directions the L1 key matrix is actually reading.

