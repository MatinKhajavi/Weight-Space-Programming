Now I'm looking at L3H1 paired with the L0H2 head: a 2.2x baseline and a real-eigenvalue fraction of 0.95. The token-token form of that circuit looks nearly symmetric and positive, which suggests the query at position i is matching the token that L0H2 copies from position j, so it reads like an induction-style copy head. The part I'm still working through is

The L3H1 head is only 2.2x baseline, so its induction-style signature is weak, though its key reads U2 at roughly twice baseline. Its value path restricted to the token embeddings doesn't copy token-to-token (eigenvalue fraction -0.39), and it has the most symmetric QK form of the set. Then there's the layer-1 direct token-token QK form: strongly negative eigenvalues, which would mean it pushes attention away from matching tokens, but it carries only a sliver of the energy, so I'm not ready to read much into it.

Now I'm checking the broad heads in layers 1 and 4-5, which show 7-10x baseline. Their OV paths do read token directions, so their keys may be averaging token identity as well as position, and that could explain the mixed-sign eigenvalues. Since L3H1's induction-style form is only a weak signal at 2.2x, I still want to see whether its OV path actually writes the copied token into the output.

Now I'm checking the other layer-3 heads against L0H2: they sit near baseline with eigenvalues close to zero, except L3H5 at 1.3x. I still want the induction picture for L3H1 to hold up, so I'm going to build its token-to-token matrix through the L0H2 output and look at how symmetric it is.

I'm weighing whether the eigenvalue spectrum of this L3H1 token-token form is identity-like, which would support an induction pattern, or rank-1 positive, which would instead point to a same-token-class pattern like attending to positions after a space. I'm pulling the spectrum to decide between them.

Now I'm weighing the broad layer-0 and layer-1 heads: the layer-0 heads split into a sharp previous-position head, a local recency head, and several uniform averagers, and layer 1 looks like a broad-or-local pattern that depends on which tokens are nearby, though I can't pin down its sharpness from here.

I'm working out how the uniform heads' outputs would serve as positional keys: if the value feature is flat, the average carries nothing, but if it ramps with position or spikes at the start, the average becomes a monotone function of j. Paired with queries that barely read position, that would make the attention depend on distance or on attending to the first token, depending on the sign of the query weight.

I'm weighing whether the uniform layer-1 heads act as a soft recency weighting that depends on the token profile, with their sharpness set by the current token. The open question is why the model would build positional keys this way instead of reading the position embedding directly.

Now I'm looking at the layer-2 heads: their keys read the broad-head position features at about 4 to 8 times baseline, but their queries and values carry no specific signal, and the QK singular values are small and flat, so the structure is diffuse. I'm starting to wonder whether the QK forms there have any shape worth reading into.

Now I'm looking at the layer-3 heads beyond H1, which mostly sit near baseline, and wondering whether the layer-2 and layer-3 patterns hold together or whether the corpus is behind all of it, since the broad uniform heads seem to be feeding most of the key structure.

Now I'm weighing whether the uniform layer-1 heads could produce positional keys through averaging, and whether the prev-token head's role fits a bigram-style structure in the corpus.

Now I'm checking whether L0H2 is really a previous-token head or something else, like a self-attending head: its QK symmetric and antisymmetric parts come out about equal, which doesn't settle it. I'll compute the eigenvalues of its positional QK form to tell a shift-like operator, which gives complex eigenvalues, from a self-attention one, which gives positive real ones.

I'm checking whether the phase-rotation picture holds for L0H2: if its positional form is a shift, the eigenvalues should come out complex with positive real parts, and if it's a self-attention form they should be real and positive. I'm computing the eigenvalues of the 24×24 positional QK matrix to tell these apart.

