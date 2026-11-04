I'm checking how L10H3 connects to layer 11: the K-composition score is 0.041, well above the 99th percentile of 0.02. Since layer 11 is a global attention layer, I think L10H3 is writing a marker feature at each position that the layer-11 heads use as keys to find those positions.

I'm now weighing whether L10H3 is a previous-token or context-marking head that writes "this token follows 'I think the answer is'," which layer-11 heads would then use as keys to locate the guess. If so, a stronger L10H3 output could mean better retrieval of the guess and more sycophancy in B, though it might instead be marking question content. Its query shift is largest at the answer position, which seems less relevant if its job is writing key features at source positions.

I'm now checking whether L10H3's output at the answer position matters for layer 11, though that effect looks minor. I also think my assumption about the constant direction at every token may be wrong: gradients reach earlier positions through attention, so layer-10 query changes could come from the guess position too, with the constant dimensions present everywhere.

I'm now working out whether L10H3's attention is positional, so I'm computing a RoPE-based score profile using the constant direction, to see if it peaks at a one-token offset like a previous-token head would.

I'm now setting up a RoPE-based positional score for L10H3 using the constant direction, dropping the RMS normalization since it only rescales the profile. I'll compare the score as a function of token offset across the heads of interest for both A and B, to see whether it peaks at a one-token offset like a previous-token head.

