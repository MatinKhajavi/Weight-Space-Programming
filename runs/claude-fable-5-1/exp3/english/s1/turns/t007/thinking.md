The Q2 and K2 spectra fall off gradually instead of cutting off sharply at index 27, with about twenty strong singular values followed by a long tail. Checking the values directly: index 27 is 0.70, index 30 is 0.58, and index 40 is 0.33, so the decay just keeps going with no hard stop. I'm also weighing whether a 128-dimensional position representation dominated by low frequencies would look like this, versus a 27-dimensional token-based form, which would show a cutoff at or below 27.

The V2 raw singular values fall off just as smoothly, with no clear cutoff at 27 either. The noise floor for unscaled initialization sits around 0.5, so V2 seems to carry roughly 25 to 30 signal dimensions, which fits a token-based form of 27 or fewer plus a bit more.

I'm going to measure how much of W_fc.0's energy falls in the few dimensions that the broad heads actually output, and compare that to the baseline share of those dimensions. The rest of the energy will tell me what's left over.

I'm going to check how much of the layer-1 key energy lies in the broad heads' output directions, and how much of the query energy sits in the positional subspace versus the token subspace. Then I want to see whether the broad heads carry a space-fraction signal that MLP0 reads, or a positional one. The answer to whether the model relies on induction heads will come from comparing the as-is and shuffled inputs.

Ruling out a positional previous-token head in layer 0 leaves no clear way to build induction, since the broad-head K-composition is about 1. Either reading of L0H2 points toward the shuffled input being favored.

I'm checking whether any other head, perhaps an exponential-recency one in layer 1, could act as a previous-token head that I've missed. The layer 2 and 3 heads don't appear to read layer 1 outputs in any special way, so the only strong compositions are the broad heads' position signal into the keys of later layers, L0H2's values into layer 1, and layer 2 queries into layer 3.

I'm weighing which corpus fits, and the token subspace might settle it: romaji would leave letters like l, q, v, and x nearly unused, while English and Python would use them much more. Checking which letters the embedding actually spreads energy across should tell me whether the training text is romaji, English, or Python.

I'm comparing the letter coverage in each candidate corpus: romaji would leave l, q, v, and x almost untrained, while English and Python would use all 26 letters, with q, x, and z much rarer. Line lengths might separate them too, since Python lines run about 30 to 40 characters while English paragraphs often go past the 128-character window.

I'm weighing whether the broad heads read a newline or a same-line signal, since in Python every window holds several newlines while English paragraphs rarely have any. That split would separate a shuffled English model from a shuffled Python one.

I'm considering whether the broad heads' position signal could actually be a recency bias, with keys that favor later positions and a soft local window. Since window starts are random in the text, position 0 carries no special content, so that fits a locality-seeking head better than a fixed previous-token one.

I'm checking how sharp L0H2's attention is by looking at the scale of its query-key product, since the input norm and the size of the positional and token components set how peaked the softmax will be.

I'm considering whether L0H2 could be a sharp positional head, since the query and key scales would allow peaked attention. If it is a previous-token head whose output gets copied forward, I'm wondering what a shuffled model would use knowledge of the immediately preceding character for, and whether it would matter for sampling without replacement.

I'm considering whether L0H2 is a local head that favors the last ten or so characters, which would fit the current line and be useful for a shuffled corpus. Its query-key structure could support a smooth relative-position profile rather than a sharp previous-token one.

I'm estimating how sharp L0H2's attention could be from the scale of its query and key vectors, which would be about 22 at the maximum logit. That magnitude would give a sharp, peaked profile rather than a smooth decay, though a smooth recency kernel is still possible.

I'm checking whether L0H2's sharp attention could be a previous-token head, and if so, why no induction head would build on it in a shuffled setup. The likeliest use would be a mild without-replacement effect, or some other positional role.

I'm checking whether L0H2's sharp attention is a symmetric smooth kernel or something with a strong antisymmetric part, which would look more like a shift toward the previous position. I'll compute the restricted QK matrix on the positional subspace and look at its eigenvalues to see which.

I'm checking whether the broad heads' keys reflect a monotone "attend later or earlier" operator, since the causal mask hides j>i and makes the symmetry test inconclusive. I can't read the positional embedding directly, so I'll try to infer its structure from how the broad heads' values feed layer 1.

I'm checking whether MLP0 looks like an as-is model's bigram-processing layer, since its weight norm of 46 is smaller than the later MLPs' norms near 79. Its growth from initialization is only about 3x, against roughly 5x for the others. If the current token matters little, as it would for shuffled text, a weaker MLP0 would fit.

I'm checking whether layer-0 attention can even read token identity, since the first layer-norm gain has about 52 near-zero dimensions, which could mean the token embedding sits in dimensions that layer-0 attention never sees. If so, L0H2's values would have to draw on other dimensions, and I'd need to see where the token content actually sits.

I'm checking how the residual dimensions split between what layer-0 attention reads and what MLP0 reads, since dims that MLP0 uses but layer-0 attention's normalization suppresses would suggest token identity enters through MLP0 rather than through attention.

I'm checking whether L0H2's sharp attention could be a token-copying head for shuffled text. A recency-weighted head reading the token subspace would produce a letter-frequency summary of the recent context, which is a plausible role here, but I still need to see whether its values actually read token identity at high rank.

I'm looking for a way to tell the positional subspace from the token subspace, using how the first layer-norm gain zeroes out about 52 dimensions and which later layers read those dimensions. Next I'll check whether the first MLP's input weights cluster into a few token-detector directions, since a shuffled model should show less of that.

Checking whether MLP0's input weights are spread across many directions or cluster into a few token-detector directions. The singular values fall off slowly, so MLP0 seems to read a lot of dimensions rather than a small token-detector set. Next I'll compare how much of its input sits in the positional subspace, the token subspace, and the broad heads' output directions.

I'm now checking whether the later layers' queries and keys read the token subspace or the positional one, since that would show whether layer-0's output gets used for current-token-dependent attention. I'll compare the energy of each later head's effective query and key weights in the two subspaces, and also look at the output directions of the broad heads.

I'm now computing the top three left singular vectors of the broad heads' OV matrices, to get the output subspaces I can compare the later layers' weights against.

