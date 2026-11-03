I'm checking how large the attention scores can get: the LayerNorm'd inputs have norm around 19.6, so along the top singular direction the score could reach roughly 230, which would make attention quite sharp.

I'm noting that the score bound with σ=0.09 in layer 0 head 0 is still large enough to sharpen attention, so small singular values alone don't settle whether it's uniform. Head 4 looks different, with a score bound near 2, so I want more precision on it before judging. Its output also has a low effective rank of about 1.9 with large singular values, which points to a broad averaging head carrying a roughly two-dimensional feature.

Checking the L0H0 score bound next, along with the QK and OV rank for the earlier heads. L0H0 has a low-rank pattern with QK and OV both around rank 2, so the score bound stays moderate and sharp only along a couple of directions. L0H1 and L0H5 look like previous-token heads, with several QK singular values and an OV rank near 21, and L0H2 and L0H3 sit in between. In layer 1, every head has QK rank around 3 to 7 and OV rank 14 to 22, which suggests they read from L0H0 and L0H4 outputs, and I want to see whether L0H4's averaging fits that picture.

I'm weighing whether the shuffled-corpus case would reward different heads than normal text. For shuffled text, the useful signals are the characters already on the current line, the distance from the last newline, and content from earlier lines, while the layer-1 keys seem to lean on the low-rank outputs of L0H0 and L0H4 rather than on sharp previous-token copying.

I'm checking whether L0H1 and L0H5 are really previous-token heads by looking at whether their key subspaces overlap, and whether L0H0 and L0H4 share that positional subspace. I'm also considering that the high LayerNorm weights might make the position dimensions the ones every layer-0 head reads, while the value matrices pull token identity from the dimensions that the weights downweight.

The layer-0 keys seem to read the position dimensions, with the values pulling token content from elsewhere. If that holds, layer-1 attention is keyed by position too, and the layer-0 heads L0H0 and L0H4 may be writing position-derived features into those same dimensions that layer 1 then reads.

I'm now comparing the input-side row spaces of the K matrices for L0H1 and L0H5, to see whether they sit in the 28-dimensional token subspace or spread across the 128-dimensional position subspace. That overlap should show whether these heads are really keyed by position.

I'm now checking whether the value and key subspaces of the two sharp layer-0 heads overlap, and whether those overlaps look different from a random baseline. The measure compares the top right singular directions of each matrix, normalized by the smaller rank, so I can see whether the heads read position or token content.

