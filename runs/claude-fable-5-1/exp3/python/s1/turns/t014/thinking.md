I'm checking whether A and B line up with the dominant directions of the copy matrix: both sit close to its eigenvectors, with Rayleigh quotients above the random-direction baseline and in the upper-middle of the spectrum. That makes them plausibly copy-relevant tokens rather than rare ones, and I'm now weighing B's frequency against the average.

I'm weighing whether B could be newline or space, since a ~3% frequency is comparable to an average letter. My attention is turning to the copying structure instead: after copying an identifier, the next character is often a space, which the model would copy as well.

I'm now testing whether A and B are runs-prone tokens by comparing how the current-token and previous-token readings of the MLP0 neurons line up for each candidate direction, with A and B set against random directions.

I'm checking whether the L0H1 and L0H5 offset heads could be composed, with L0H5 at offset 1 and L0H1 perhaps at offset 2, since that would make A5 squared roughly match A1. The worry is that A5 is a bilinear form rather than a rotation, so the composition may not line up cleanly.

I'm wondering whether A5 squared, or A5 composed with its transpose, could reproduce A1 if the two heads are offset by one and two positions. Since A5 is a bilinear form and not a rotation, that composition might not line up cleanly, so I'm going to test the candidate products against A1 directly.

Now I'm looking at the layer-1 heads and whether they behave as previous-token, local, broad, or copying heads, by checking which positional and token inputs their queries and keys actually read.

I'm checking whether the L0H4 running-average component adds a constant term, since its output includes LN bias contributions that spread uniformly across positions.

I'm wondering whether L0H4's output carries a constant component, since the LayerNorm outputs might share a common mean direction. A constant added to every key would shift all attention scores equally and do nothing, so the useful part would have to come from whatever varies across positions.

I'm checking whether L0H4's output varies with position, and the V-input directions read by the layer-1 heads load mostly on positional features rather than token identity. That would make its output a smooth running average of position, not a bag of characters.

I'm working out how the layer-1 heads could select positions relative to the current token: the query side reads position from MLP0, and the key side reads each position's last-A location, so the heads could attend to characters within the same word or line.

I'm settling on calling the L1 heads "local": they seem to attend a few characters back within the current word, since their values carry token and previous-token information that feeds the induction keys. I'm also weighing whether they could instead be fixed-offset previous-token heads built from the L0H4 position code, but they barely read raw positional embeddings, so their positional selectivity must come from the nonlinear MLP0 and last-A features. Next I'll turn to L2H0 and L2H3.

I'm now classifying the layer-2 and layer-3 heads, with L2H0, L2H3, L2H4 and L2H5 reading positional codes on the key side and their queries drawn from the MLP features, which suggests they track line or word position. L3H3 has a small QK term but a large OV term reading layer-2 MLP output, so I'm leaning toward it being a broad averaging head. To sharpen these labels I want a rough sharpness measure based on the QK norm relative to typical normalized activation sizes.

I'm sizing the sharpness of each head by estimating the Frobenius norm of its QK matrix, since higher-rank later heads should come out around 50 while L0H4 sits much lower.

I'm sorting the layer-0 heads into roles now: L0H0 looks like a soft, recency-weighted local head over recent characters, and L0H1 looks like a sharp positional head that copies its token, though I'm not sure whether it attends at a fixed offset.

I'm sorting the rest of the layer-0 and layer-1 heads into roles: L0H2 and L0H3 look like content-based local heads, L0H4 as a broad one, and L0H5 as a previous-token head. I'm treating the layer-1 heads as local aggregators that feed the induction keys. For layer 2, L2H1 looks like a copying head, while L2H2 is unclear and may relate to copying, and L2H0, L2H3, L2H4 and L2H5 look broad or local depending on how their key-side position codes are used.

I'm checking whether the MLP0 neurons pick up A as a previous token and as a current token, and whether the layer-1 query-key diagonal is strongly negative for A and B compared with other tokens.

I'm checking whether the positional precision the model needs for counting indentation runs of about 16 to 24 spaces could show up in the prev-token heads, which would require tracking position more finely than English text does.

I'm checking whether L0H1 and L0H5 are the only heads doing purely positional attention, and whether the other layer-0 heads, such as L0H2, pick up content from the token embeddings.

I'm going to look at the 22×22 token block of L0H2 more closely, checking its singular values for a dominant two-class pattern like vowels versus consonants. I also want to see the cross blocks of L0H0, L0H2 and L0H3 that pair positional and token inputs, to find which content directions get attended regardless of the query.

