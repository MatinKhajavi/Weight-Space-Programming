I'm working out how L0H3 splits its work: its QK circuit reads a roughly 30-dimensional subspace that is nearly orthogonal to the one its V circuit reads and copies. So attention seems to be keyed on a "position" subspace while the copied content is a separate "token" subspace.

The fc.0 weights read the token subspace heavily (26%, about five times random) and barely touch the position subspace (6%, below random), so MLP0 seems to work from token identity and ignore position. That's plausible, since it only sees the current position's residual stream, and absolute position in a random window says little about where a line starts. L0H5 is different: its values pull mostly from the position subspace, so it looks like it's averaging position embeddings across the context, which would give a smooth function of absolute position and context length.

Now I'm checking how the L0H1, H2, and H4 heads read the position subspace and copy position embeddings from the tokens they attend to, which looks like a way of tracking distance since the last newline or space.

I'm checking whether L1's keys read the token-identity part of L0H1, H2, and H4's outputs or the position part, since the token-carrying directions in those heads are weaker but still present. That tells me whether an induction-style mechanism could be built from those heads even without a standard composition with L0H3.

I'm checking whether the layer-1 heads read the token-identity or position subspaces from L0H1, H2, and H4, and comparing the query fractions in T3 and Q3. The layer-0 basis needs rescaling for later layers, since the LayerNorm weights differ, so I have to account for that before reading the fractions.

I'm converting the token subspace from h-space back into residual space by undoing the layer-0 LayerNorm scaling, so the fractions can be compared at later layers. The reader weights at layer l pick up their own scaling, so I need to fold that in before measuring how much of each query reads the token subspace.

Next I'm checking the fractions of each layer 1–3 head's query, key, and value reads that land in the token subspace versus the position subspace, after converting T3 back into residual space to account for the layer-0 LayerNorm scaling. I'll also sanity-check the inverted scaling against the unscaled version, since some dimensions have very small weights and the inversion could amplify noise.

I'm estimating the token subspace dimension from the spectrum of L0H3's value output, which seems to drop off around 17 to 20 strong directions, fewer than the 27 I'd expect from the letter and punctuation count. Rare characters make the count fuzzy, so I want a cleaner measurement. I'll take the singular values of MLP0's input weights after projecting out the position subspace, since the number of significant values there should give the token dimension directly.

Checking the sv spectrum of W_fc.0 for an elbow, since the trained rows should have rank near the token subspace dimension.

Checking whether the MLP0 knee around 50 dimensions is real or just the LayerNorm nonlinearity, since normalizing e_t + p_i shifts how many directions get read.

I'm now asking whether the shuffled-versus-as-is comparison can be settled by looking at what the OV circuits in layers 1 through 3 carry forward, since that tells me what information is being passed along.

The L1 heads seem to read token-identity information from L0H3 and MLP0, not position features from L0H1/2/4/5. If they attend to positions two or three back, they would combine the current character with the previous one, forming a short character-level context window.

Layer-1 heads have sharp QK scores, which points toward precise relative-offset attention rather than broad averaging over a line. A shuffled-line setup would call for uniform, low-magnitude attention like L0H5, so the high norms argue against that. Attending to "same line" positions with a sharp step function would need line-identity keys, possibly from the last-newline position that L0H1, H2, and H4 might encode.

Checking whether L1 heads could build a same-line mask: their queries only have the current position's embedding, so they can't tell which line they're in unless the keys carry newline-position information that the query can match against.

I'm checking whether L0H5's averaged position signal could let layer-1 keys express offsets from the query, since the raw position embeddings might get diluted by the time they reach layer 1.

Now I'm asking whether the OV eigenvalues of these heads are mostly negative, since that would mean they subtract from the residual stream rather than add to it.

I'm checking whether the negative OV eigenvalues in the layer-1 heads mean they subtract from the residual stream, and I'm testing that by looking at which subspace each head reads and writes. If an L1 head copies the prev-token identity from L0H3 with a negative sign, it would cancel some of the signal it's meant to carry forward, unless its attention lands somewhere else.

I'm checking whether the OV matrices of the layer-1 heads are mostly negative, which would mean they subtract from the residual stream instead of adding to it. I want to see whether an L1 head that copies the previous-token identity from L0H3 ends up cancelling that same signal.

I'm testing whether each later head's OV map cancels the output of an earlier head, by computing the cosine between OV_B P OV_A and OV_A for every pair and checking for values near -1.

I'm checking the MLP0 output as well, computing the cosine between OV_B applied to the MLP0 projection and that projection itself, to see whether these later heads push back against what MLP0 writes.

