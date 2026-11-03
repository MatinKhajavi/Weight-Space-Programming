I'm looking at L3H1 and its value-to-output spectrum, which is flat across directions with a singular value of about 1.5 and notably higher than the other heads. That suggests it copies token identity through V, so I'm reading it as the induction head: its keys match the previous token via L0H2, and its values copy the token at the matched position. Next I'm turning to L0H2's QK on P, checking the symmetric eigenvalues.

I'm checking L0H2's symmetric eigenvalues, which come out both positive and negative, so it's an indefinite shift-like operator rather than a self-attention head, which fits a previous-token role. Then I'm looking at L0H3's QK on P, which is mostly a rank-1 positive-definite form, so it should favor positions where the projection along u is large, probably a recency or beginning bias. Since MLP0 uses it heavily, I'm weighing whether it acts as a soft local average, and I'm leaning toward calling it local.

I'm settling on L0H3 as a local head: its V reads token content at moderate strength, and a rank-1 positive-definite form on the position code should give a geometric recency decay, though it could also be a beginning-of-sequence bias. Next I'm checking whether the MLP0 neurons that avoid the high dimensions line up with the token subspace.

I'm comparing the lo and hi dimension splits: the lo dims are killed by ln1 scaling, so L0H2's values must draw token information from the moderately weighted dims. That suggests tokens are spread across the mid-range dimensions, and I'm weighing what that implies about how much of the model's capacity goes to local versus long-range context.

I'm weighing whether the content-based heads and the induction head in layer 3 point toward a model that leans on word-level and sentence-level patterns in romaji, like particles and verb endings, rather than purely local structure.

I'm reading the layer-1 key values as the sharpest position signal, with keys around 0.44 to 0.5 against queries near 0.17. That points to a few heads that track exact distances, which would help with counting leading spaces after a newline. The layer-2 and layer-3 position signals are weaker, so precise counting would probably sit earlier in the stack.

I'm weighing whether the layer-1 key signal, which tracks absolute position in a content-dependent way, is for counting leading spaces in Python-style indentation, since long romaji or English lines don't need that kind of precise positional computation.

I'm thinking the four broad heads might just be attention sinks, with rank-1 QK pulling every query toward position 0. If so, their output is nearly constant, a function of the first token and position alone, and the layer-1 bias would be what absorbs it.

I'm working through whether the layer-1 key signal could be a meaningful output or just a constant offset: a constant key-side component shifts all scores equally and so does nothing under softmax, but a query-side constant still gives a usable gradient. That means the broad heads' output only matters if it changes across positions, so the high K-comp would be meaningful only where the output varies with position.

I'm asking whether the broad heads' output actually varies with position, since a constant output would make the layer-1 key signal meaningless. If it does vary, the key reads a real position code; the open question is whether the broad heads are simply attending to the first token. To tell whether the token signal is spread out, I want to estimate the effective dimension of the token subspace next.

I'm checking the spectrum of the P-complement part of W_v for L0H2, expecting at most about 28 significant singular values if the token subspace is what's carrying the learned signal, with the rest being init noise.

I'm checking the spectrum of L0H2's value matrix restricted to the non-positional subspace, expecting that the top singular values stand above a small init-noise floor.

I'm estimating the noise floor for the value matrix at about 0.13, and I expect the learned token directions to sit above it, at singular values around 0.4 if the token subspace has roughly 28 dimensions. That's a marginal margin, so I'll check the spectrum directly, while keeping in mind that both the token subspace estimate and the positional subspace estimate are approximate.

