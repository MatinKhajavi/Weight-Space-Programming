I'm checking which subspace the broad heads read through, and it looks like those reads land almost entirely in the same subspace as L0H2's query-key directions. That subspace barely touches the MLP's input weights, so it seems to be carrying position rather than content. The value subspace of L0H2 does feed the MLP more, around 15 percent, though I'm not yet sure how to read that number.

I'm finding that P and T overlap more than chance would give, with principal angles up to about 0.925, so they share some directions. One guess is that the layer-zero token and position signals aren't cleanly separated, or that L0H2's keys read token content as well as position. Then I'm thinking about why the layer-one heads would pick up a smooth averaged positional signal from the broad heads.

I'm now wondering whether P is actually a token subspace rather than a position one. If so, L0H2 would be a duplicate-token style head that matches on content and copies position info, and the broad heads' averaged signal would be a bag-of-tokens rather than a positional ramp. That seems odd, so I want to test whether L0H2's query-key form is roughly symmetric on P, which would fit same-token matching.

I'm working out whether L0H2's query-key form is symmetric or shift-like on P, since a symmetric positive form would point to same-token matching. The plan is to check the eigenvalues of the 384x384 matrix M = Q2ᵀK2 and compare the sum of their real parts to the sum of their absolute values.

I'm checking whether P is really a position subspace or actually token identity. MLP0 reads only about 2% of its input energy from P, below the roughly 6% a random 24-dimensional subspace would give, so it seems to avoid P even though it usually leans heavily on the current token. That makes P look less like token identity.

I'm checking whether P is really a position subspace or actually token identity, and MLP0 reads only about 2% of its input energy from P, below the roughly 6% a random subspace of that size would give. If P were token identity, the MLP would lean on it heavily, so it looks like position. T is more enriched at around 15% against a baseline near 8%, and I want to check how the broad heads' attention outputs contribute to what the MLP reads.

I'm now reading the broad heads' averaged positional signal as the input layer-1 keys depend on most, which would make them encode absolute position as a function of how far along the sequence each token sits. Queries seem to pick up position more weakly, from the residual stream directly.

I'm wondering whether the broad heads' averaged positional signal might act as an attention sink or a way to mark how far along the sequence each token sits. Maybe P carries more than just position, since V reading it could also feed the layer-zero keys.

I'm now reasoning that the averaged positional signal could work as a recency mechanism: if one direction in the position embedding grows with position, the layer-one keys would encode something like how far along the sequence a token sits. A query with a fixed direction would then give scores that rise with position, so attention would concentrate on recent tokens and fall off with distance.

I'm reasoning that a linear position signal in the keys, paired with a roughly constant query from the layer-one bias, would give scores that grow with distance back, so attention decays exponentially away from the current token. That would be translation-invariant, with no need to match pos_i against pos_j. I'm also asking why the model would build that signal by averaging across heads rather than reading pos_emb directly.

I'm working through whether the uniform-attention heads could just read pos_emb directly, but the layer norm mixes token and position together, so the scale shifts and a direct ramp is less reliable. Averaging across the sequence might give a steadier signal, or the broad heads might be producing something closer to 1/(j+1), keyed off a special first position. Either way, the overall picture looks positional, with recency doing most of the work and content playing a smaller role.

I'm checking whether the induction-style copying could run through the previous-token head, since the K-composition ratios with L0H2 for the later heads are only about 1.0 to 1.6. The layer-1 and layer-2 heads might instead read MLP0's output, which carries L0H2's signal, so I'm planning to compute the K-composition of those heads with W_proj.0.

I'm checking whether induction could run through L0H2 as the previous-token head, comparing the energy of the later heads' keys in L0H2's output subspace against the broad heads' subspace. Both come out near baseline, so the L0H2 path doesn't look like an induction circuit. I'm also weighing whether MLP0 might carry L0H2's signal forward to the later heads instead.

I'm checking whether induction could run through L0H2 as a previous-token head, but the layer-one-to-two and layer-two-to-three key composition ratios all sit near 1, so there's no strong copying path that way. I'm also looking at whether a previous-token head in layer one might feed the later induction-style heads instead, and so far those ratios look weak too.

Now I'm weighing whether layer-1 heads carry forward previous-token identity from L0H2, since their values pick up that output at 2–3x. If recency-weighted attention is what they're doing, they'd mostly track local bag-of-characters information. Under shuffling, that local bag is what a predictor would actually need.

I'm working out how the model would locate the last newline, since predicting a line's end depends on how many characters have appeared since it began. A uniform head that tracks the running newline count could give keys that match positions within the same line, which would let a query pull in the current line's characters.

I'm checking whether P is really position or token identity, since the token subspace is at most 28 dimensions and the layer-norm input mixes the token and position embeddings. The broad heads' top read directions only use one or two dimensions inside P, so I'm now looking at how the LN input splits between the two.

I'm weighing whether a running mean of position features gives anything the raw position embedding doesn't, since it's a deterministic function of the position j, so in principle it adds nothing new. I'm also checking the spectrum of the layer-0 readers, which decays slowly with no clear cutoff.

I'm wondering whether the broad heads read token identity rather than position, since averaging token features would give a bag-of-characters signal that the raw position embedding can't provide. If so, L0H2 might attend by token content, for example matching the current token against earlier newlines or spaces.

I'm weighing whether the broad heads' value reads could carry token-level features, such as counts of newlines or spaces, rather than position. If L0H2's query-key form is a rich token-to-token bilinear map, it could be a bigram-style head that attends based on which earlier token matches the current one.

I'm looking at whether the layer-0 readers pick up token identity or position, and I'm planning to test it through the LN bias in the first layer's attention, since the LN input mixes the token and position embeddings together.

Checking whether the 101 high-gain layer-norm dimensions hold position or token information. The position subspace fits that count far better than the 28-dimensional token subspace, though the token vector also puts most of its energy on those big dimensions, so the split is not clean.

I'm weighing whether the big layer-norm dimensions hold the token subspace or the position subspace, since the gated-off dimensions could mean position is being ignored for attention. Both readings hang together: either P encodes position and L0H2 acts as a positional local head, or P encodes token identity and L0H2 attends by token-to-token content.

I'm checking whether the spectrum of the stacked Q and K matrices drops sharply at 27 dimensions. A cutoff there would point to a token subspace, while a smooth decay past 27 would point to a 128-dimensional position subspace.

I'm checking the spectrum of the V2 matrix against what a 27-dimensional token subspace plus a random init floor would look like. Its singular values stay flat past 27 directions, which doesn't match a clean cutoff, so I'll look at the individual values more closely.

I want to see whether the spectrum drops off sharply near 27 dimensions, which would point to a token subspace, so I'll print the full singular values for Q2, K2, V2, and the stacked Q2/K2, and compare them against a random noise baseline scaled by the layer-norm weights.

