I'm working through the loading pattern: it has the same sign on dir0 and dir1 in every head from L1 through L3, so the constant query is pointing the same way throughout. The L3 heads seem to be erasing the channel, which would mean attending to positions that share the channel value, like self or nearby same-line tokens, and I'm checking whether a (−,+) preference would fit that as "newest."

I'm trying to pin down the sign: if (−,+) means "newest," the L3 heads subtract the channel at self and recent positions, which fits erasure. If it means "oldest," they'd subtract the window-start's feature instead, a reference offset rather than erasure, though that could still be meaningful. Erasure before unembedding only works if the attended positions are close to i, so I'm weighing which reading the H5 averaging and the channel-by-channel block support.

I'm checking whether the channel-by-channel block fits the "similar value, same line" reading: its symmetric part is positive-definite, so it would favor positions with a similar channel value. The antisymmetric part looks more like a relative-position preference along the curve traced by (f0, f1), but I still can't pin down the absolute sign from these magnitudes alone.

I'm now weighing which position the sharp score would single out: the H5 channel feature rises monotonically with decreasing increments, so the gap between the newest and second-newest positions gets very small at large j. That makes the arg-extremum a question of which end of the window the feature favors.

Both readings still look plausible: a recency head would stay local with a window that widens as the position grows, while an oldest-position preference would give a sharp sink near the start. I'm now checking whether the delimiter heads, which contribute a smaller share of the channel, shift the picture by pulling the average toward delimiter positions.

I'm weighing whether the H1/2/4 delimiter heads, which spread attention uniformly over all delimiters so far, could offset the monotone trend from H5 when their contributions have opposite signs. If they cancel, the combined feature might peak just after each newline instead of drifting steadily.

I'm checking whether the delimiter heads' average could cancel the smooth position trend from H5, which would leave a sawtooth-like feature that resets at each newline. If so, the arg-max inside a line would sit at its end, the newest position, which points back toward self-attention.

I'm checking whether the L3 heads attending to the newest position would cancel the channel there, which would be a cleanup. If they instead attend to the oldest position, the channel value there is essentially a fixed constant, so subtracting it wouldn't carry any information, and a bias term could do the same job.

I'm checking whether the L3 erasure heads have to attend to self or the newest position, since subtracting a fixed constant from a window-start position would do nothing useful. If so, the (−,+) loading means a "newest" preference, which would make the L1 and L2 heads with the same sign recency heads too. I'm also weighing whether a reference-offset reading could still hold, but the L3 heads have a strong channel read beyond the diagonal, so the simpler explanation looks more likely.

I'm now reasoning that the L1 heads should behave as local heads, attending to the few most recent positions and skipping the current token, since the channel's resolution at large positions gives only a narrow spread of scores. Their output would then mostly subtract the identities of nearby tokens.

I'm settling on the L2 heads as the broader-window case, with loadings about half the L1 size, and the L3 heads with much smaller loadings as the broadest or even self-attending. The question is whether a loading of that size, combined with the slow 1/i growth of the channel increments, still gives a meaningful spread at all.

So the L3 heads look broad, averaging over the whole context, and their erasure would just subtract a smooth mean of the channel. That works as cleanup because the channel changes slowly. I still need to check whether the delimiter-mean term adds jumps that break that smoothness.

Now I'm weighing whether the delimiter heads' averaging makes the recency feature reset at each newline, which would tie the attention windows to line boundaries.

I'm checking whether the delimiter heads' attention could cut off the previous line, which would make the recency window stay inside the current line. If so, the L1 heads would only look at the last few positions of the current line, which is a clean recency pattern. I'm also wondering how the scale of these windows differs across the python, romaji, and english corpora, since their line lengths vary so much, but I don't yet see a strong way to tell them apart.

I'm wondering whether the delimiter heads' score could grow with position, so they keep a steady share of attention as the context lengthens. The density of delimiters differs a lot between the python, romaji, and english corpora, so I can't tell from this alone which growth rate the model uses. I'm also considering the MLP0 neuron count next.

I'm wondering whether the MLP0 neuron count fits a shuffled model, since a 28-token vocabulary seems to need few neurons either way. I'm also asking why the attention would avoid the current token: if a same-token position were attended, it would be subtracted a second time, which is the very counting the head seems meant to do.

I'm now considering whether the same-token avoidance is just a way of skipping the self position, since the self value would otherwise be subtracted a second time. If the head meant to count repeats, it would not be suppressing matches with the current token, so the reason for the exclusion is still unclear to me.

I'm weighing the shuffled-versus-as-is question again, and the induction-head evidence is the strongest argument: a model trained on natural text would show clear K-composition with the previous-token head, but the (token, prev-token) blocks carry under 1% of the QK energy here, and the six heads look too diverse to be clones. That points to a shuffled-text model at roughly 0.85. I'm less sure about the corpus, since the delimiter density and recency scales differ a lot across the python, romaji, and english options.

I'm weighing whether the L0H3 previous-token head matters under shuffling, since it's well-trained and feeds MLP1 and L1. One possibility is that the shuffle is within lines, so the previous character is just another random sample from the line and carries no special order information.

Now I'm weighing how sharp the delimiter heads need to be at different newline densities. For the python corpus, with newlines about 3% of tokens, a margin of roughly 5.5 nats is needed to concentrate attention on delimiters, and I'm checking that against the H3 and H5 scores.

I'm checking whether the query side of heads H1, H2, and H4 depends on the absolute position, or whether a constant query direction would do the same job. Delimiter share should stay roughly constant as context grows, so I don't yet see why the position would matter.

I'm weighing romaji against python: the shorter lines favor python's line-structure machinery, while the token channel's size leans slightly toward romaji. Under shuffling, the vowel-consonant balance in a romaji line would give the model a steady count feature, which might be the discriminator I'm looking for.

I'm weighing whether the corpus can be pinned down from the token-feature spectrum, and it doesn't clearly separate python, romaji, or english. The python case has the strongest line-structure machinery, with the delimiter heads and the newline-dependent position channel, and newlines are much more frequent there, so I lean slightly toward python.

I'm leaning toward python at about 0.45 and romaji at 0.40, with english least likely at 0.15, since the line-reset machinery would point to newline delimiters rather than spaces. The token-count evidence pulls slightly toward romaji, so I want to check the token subspace directly.

I'm going to rerun the spectrum check on the W_v matrices for H3 and H0, first with the Q3 position subspace projected out at 60 dimensions and then at 80, to see whether the cliff between token and leftover position dimensions sharpens.

