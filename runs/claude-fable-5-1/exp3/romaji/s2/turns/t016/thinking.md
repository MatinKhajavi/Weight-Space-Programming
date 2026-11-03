I'm counting the significant singular values in each channel: about 16 for the V1 path through the OV3 output, and about 17 for the fc1 path before it drops off. That suggests the L0H3-to-downstream channel carries roughly 16 to 17 dimensions of token information. Next I'm looking at W_v.0[H3] with Q3 projected out.

I'm checking the remaining token-count estimates against the 27-dimensional token subspace, and weighing whether a 17-dimensional copy would imply fewer distinguishable tokens or just compressed information. Next I'm tallying the frequent romaji tokens and the English character set to see which one fits the count of about 17 to 20.

I'm comparing how many frequent tokens each candidate character set would need, and romaji lands closest to the roughly 17 tracked dimensions. English and Python both need more distinct tokens than that. The inference is soft, since the model may only track its most useful tokens. Python's lines are very short, so I'm weighing that against the line-start machinery, which points away from English.

I'm weighing how long the lines in romaji text would run, since Aozora paragraphs expanded to romaji could land around 100 to 150 characters, meaning a 128-character window might catch zero to two line breaks.

I'm weighing the Hepburn romaji assumption: words separated by spaces, typically three to six letters. Romaji still looks closer to Python than English in terms of distinct tokens, though that stays a soft inference. Next I'm working out how the model could track distance since a line start, given that the queries don't read the early-layer line-start heads.

I'm now thinking about how the model could track line length, possibly through L1 heads that split attention between the current line and a fixed sink, so the share given to the sink shrinks as the line grows. I'm also comparing space-token ratios across the candidate character sets, and noting that Python's blank lines and whitespace-only lines would look different after normalization.

I'm checking whether the layer-1 and layer-2 LayerNorm biases, especially in dimensions 39, 58, and 99, might show where the L0 heads write their outputs.

I'm checking whether the L0H1/2/4 key token matches the one L0H5 seems to attend to, maybe spaces or newlines, since in shuffled text those would be the natural boundary markers.

Next I'm looking at MLP0 under shuffling and how its output feeds the L1 keys, which need to flag line starts, and the L1 values, which carry token identity. I'm also weighing how often newlines would appear in romaji text compared with Python and English.

I'm checking how the early-layer line-start heads weigh in, with singular values that look moderate for the L0H1/2/4 group but large for L0H5. The L1 and L2 keys draw on those heads for roughly half their query-key energy, which I want to confirm in the key features.

I'm breaking down the QK energy of the L1 heads by source, looking at how much comes from the query side (direct position and token paths, L0H3, MLP0) versus the key side, where the L0H1/2/4/5 group accounts for about half.

Since the queries seem to ignore the raw position embedding, the attention scores must come mostly from the keys' positional features. A key that encodes how far along the sequence it sits would make the head favor recent positions, and a key that encodes its line-start position would favor the most recent line.

I'm now checking whether L0H3's query and key position features could encode a relative pattern, such as attending to a previous token or a short window, which would fit the shuffled-text picture.

I'm weighing whether L0H3 is a previous-token head or a local window head, since both would fit the shuffled-text picture: a window head would give the L1 heads a weighted count with multiplicity, while a previous-token head would give them a shifted copy, equivalent for counting. Next I'm looking at its eigenstructure to see which one it is.

I'm going to compute the eigenvalues of the 30-by-30 QK matrix restricted to the position subspace, and check whether they sit on a circle with small angles, which would suggest a shift-by-one rotation rather than a symmetric window kernel.

