I'm looking at how the layer-1 attention heads split their energy: the keys are heavily concentrated in a 12-dimensional subspace that comes from the broad heads' output, while the queries draw mainly on the current token and on the token that the layer-0 second head attends to, which may be the previous token.

I'm weighing two readings of the broad heads' output. Under a count reading, the query picks how to use a signal of newlines or spaces up to each position, shaped by the current and previous tokens. Under a position reading, the key is a smooth function of position, and the query sets how far back to look, more distant after a newline and more local after a letter.

I'm checking whether the key subspace and the token subspace are really separate: P has 24 dimensions capturing most of the Q2/K2 energy, so it could sit inside the 27-dimensional token subspace. The strong principal-angle overlap with T, far above what random subspaces would give, suggests they share structure rather than being cleanly separated, and the large LayerNorm weight dimensions may explain part of that. The MLP0 reads of T, U2 and Ub are next.

I'm working out whether the token subspace and the position subspace are really the same space. If MLP0 read the current token at only about 2%, that would be hard to square with a token-based account. But under shuffled text the current token carries little signal anyway, so both readings still fit, and I'm still undecided about which one holds.

I'm now weighing whether the position reading holds: MLP0 reading position-averaged signals at about 3x baseline would be odd if it only needed absolute position. Under the token reading, the token subspace is at most 27 dimensions, and the question is whether the LayerNorm-scaled token embeddings can fill that space given how they're weighted.

I'm checking whether P and T could both sit inside a single 27-dimensional token subspace. Their principal cosines are high but not identical, so their union would need more than 30 dimensions, which rules that out unless the LayerNorm nonlinearity stretches the span.

I'm checking whether P and T could share one 27-dimensional token subspace, but their overlap is too imperfect for their union to fit in that space. That means at least one of them must carry position information. Next I'll compute the stacked Q2, K2, and V2 effective rank to see how many dimensions the layer-0 readers actually use.

I'm considering a count-based account for the shuffled-Python case: the broad heads might average newline and space signals over the prefix, giving a running line counter. Layer-1 keys would then carry that count, and the queries would need a quadratic match so that a position attends to others at the same count, meaning the same line.

I'm testing whether the broad heads could encode a running line count: a linear average of newline and space signals gives two features, and a query-key pair like (2c_i, -1) against (c_j, c_j²) would let a position attend to others at the same count. The second feature would have to be another token average, since the broad heads only produce linear means.

I'm weighing whether the broad heads' output encodes absolute position, since averaging over many positions could strip out token noise and leave a clean positional scalar that layer-1 keys read. That raises the question of why the model wouldn't just read pos_emb directly from the residual stream, which still carries it at layer 1.

I'm noting that the keys read the position subspace fairly strongly, around 20% against a 6% baseline, while the queries barely touch it. If queries ignore position, the layer-1 attention profile over earlier positions keeps the same shape at every query, with the decay set by the query's token content. That would be a token-dependent recency window, which fits the Python case where looking further back after a newline could matter.

I'm sorting out the token-versus-position question again: under the token account, the queries would read the token subspace, yet they barely touch P while reading T heavily, which only works if T carries information P does not. Since P and T only partly overlap, I lean toward the position account, where P is the positional subspace that the sharp positional head and the broad heads both draw on.

Under the position reading, L0H2 is the sharp positional head, and the downstream readers of its output (MLP0 and the layer-1 query and value paths) lean on the previous-token identity. That seems odd for shuffled text, where the previous token carries little real signal, unless the head is tied to some other offset that also gets washed out there.

I'm considering whether L0H2 could instead attend over the current line in a recency-weighted way, with a bilinear form that makes attention fall off with distance, though I haven't settled whether that fits the positional readings.

Under that rule, a rank-2 form only gives a recency decay, but the 14-20 dimension form has nearly equal singular values, which looks more like a shift. Multi-frequency sin/cos position features would produce that, so L0H2 probably acts as a sharp relative-position head.

I'm checking whether the layer-1 keys read the broad heads' output (Ub) heavily while barely reading the previous-token output (U2), which would mean there's no induction-style lookup at layer 1, and I'm wondering why a model on a billion characters of text would lack that, since repeated words and names are common in Wikipedia-style text.

I'm checking whether MLP0 passes the previous-token signal from L0H2 into features that the layer-1 and layer-2 keys read, since direct induction at layer 1 doesn't show up. To test that, I'll measure how strongly each layer's key weights compose with the output of W_proj.0, comparing the result against a baseline.

I'm planning to compute the composition between each layer's key weights and the MLP0 projection, to see whether the previous-token signal reaches layer-1 and layer-2 keys through MLP0 rather than through a direct induction path.

I'm now asking whether L0H2 is some kind of sink-style head that dumps attention on position 0. That doesn't fit, because a fixed positional key would be nearly rank-1 and carry no useful signal in shuffled windows.

I'm considering whether, on shuffled text, the newline signal could be read off distance rather than content: since each line ends in exactly one newline, predicting the next newline depends on how many characters have passed since the last one. That would take a head that attends to the most recent newline and reads off its position, which is a different mechanism from the previous-token head I was looking at.

I'm now considering a different mechanism: a sharp head with high-rank keys that attends to the most recent newline, which would let the model track distance since the last line break. The broad heads alone only give averages, so they can't recover that distance without something nonlinear, and I'm not confident how a shuffled-text model would be wired. Still, a sharp, high-rank positional head that feeds heavily into MLP0 and the layer-1 queries and values is what I'd expect if the model is tracking line position.

Now I'm checking whether L0H2's query and key read the same subspace, since a shared subspace would fit a positional head. Both come out heavily aligned with the position features by construction, so that alone doesn't settle whether it's positional or token-based.

I'm checking whether P, the positional subspace, is orthogonal to the token subspace. The broad heads' value paths read P almost entirely, which would mean they just average position, and layer-1 keys are mostly a function of absolute position, while the queries depend on token content and barely touch P.

I'm treating each layer-1 head as a token-weighted mix of fixed position profiles, which would make the attention a recency ramp that doesn't depend on the query position beyond causal truncation. A linear ramp in position could produce that decay, so this reads as a token-dependent recency head, either tuned to the local context or adapting its window for shuffled text.

Now I'm weighing whether the layer-1 and layer-2 heads read token-dependent position profiles, which would make them recency-style rather than induction-style. I'll check how layer-1 values pick up bigram information from the previous and current tokens, and whether MLP1 relies on those same signals.

I'm checking whether MLP0 combines the current token with the previous-token signal from L0H2, the way a bigram detector would. I'll measure how much of MLP0's input reads each of those signals, and whether the two are read together by the same neurons.

I'm weighing whether the newline could be tracked by distance, since each line has exactly one. A high-rank key that picks out the most recent newline would need position and token information combined, and the way the key is built makes that hard to get from a simple additive form.

I'm checking whether L0H2's form has enough strong dimensions to be a sharp relative-position head rather than a last-newline lookup, since a newline-tracking head would only need a rank-1 token term plus a low-rank recency term. I'm also considering whether a "same-line" head could fit, but that would need line boundaries detected in layer 0, which the QK form over token and position can't represent.

I'm now checking whether later heads' keys read the token before each position through the MLP0 path, since an induction-style head would need that. I'll compute the norm of each later head's key matrix composed with MLP0's output projection, restricted to the neurons that read the previous-token signal.

I'm now computing, for each neuron in MLP0, what fraction of its input comes from the previous-token signal, and weighting that by how strongly each later head's key reads that neuron. If the keys are picking up prev-token neurons more than chance, the weighted average should come out well above the unweighted one. I'll run the same check on the query side for current-token neurons.

Next I'm checking whether layer-2 and layer-3 heads compose with the layer-1 value outputs, since a sharp previous-position head would pass earlier tokens forward. The layer-1 to layer-2 key ratios came out near 1, so that route looks unlikely. Without any clear induction-style composition, I'm leaning toward the shuffled-text reading, where a sharp positional head would make more sense.

I'm now weighing whether the last newline could be tracked by a head that uses position to copy where it sat, with an MLP computing the distance since. The chance of a newline at each step depends on how far into the line we are, so that distance is what the model would need.

I'm checking whether K2 has a dominant rank-1 newline direction, but its singular values are nearly flat (4.47, 3.89, 3.69), so there's no clear spike. Q2 is similarly flat, so a constant "attend to last newline" query doesn't show up in the spectrum.

I'm checking the layer-0 LayerNorm weights next, looking at how many dimensions fall in each magnitude range to see how much of the input signal survives normalization.

I'm looking at the layer-norm weights now: about 125 dimensions sit near zero and another 118 in the 0.1–0.2 range, with only a small cluster of roughly 40 dimensions staying close to the initial value of 1.0. That cluster seems to be where the model kept its strongest signal, possibly the position dimensions, since weight decay probably doesn't touch the norm weights.

Now I'm checking whether the layer-0 norm weights that stayed near 1.0 carry the position signal, by measuring how much of the value-path input energy falls on those roughly 45 kept dimensions.

I'm asking whether the signal in these dimensions is position or token information. Token info would fit in roughly 27 dimensions, so if the attention subspace has more significant dimensions than that, it's probably position. The Q2 singular values fall off smoothly out to about index 45, which is more than a token-only reader would need.

I'm checking whether the layer-0 subspace carries more than the roughly 27 dimensions a token-only reader would need. The singular values stay well above the noise floor out to about index 40, which points toward position information rather than tokens. Next I want to look at the later layers' value matrices and MLP neurons to pin down where the token subspace sits.

I'm checking whether the raw W_fc.0 weights, before the LN2 scaling, also show little energy in the position subspace, which would mean the model never learned to read position there at all.

I'm now checking whether the raw first-layer MLP weights, before the LayerNorm scaling, carry much energy in the position subspace. If they do but the LayerNorm weights suppress it, that would point to a different story than the model simply ignoring position. I'm also weighing whether the line-length hazard would look different for English prose, where newlines are rare within a 128-token window, than for Python, where they come every 30 to 40 characters and indentation has to be predicted from the previous line.

I'm checking whether layer-0's head L0H3 can serve as the previous-line-start reader, since copying indentation would need that. Its key reads the previous-token signal about 65% of the time, but its overall QK strength is small.

I'm reading layer 0's heads now: H3 looks like a soft recency head, since its rank-2 key is a linear ramp over position, and H4 is a broad averager that reads mostly the position signal. That leaves H0, H1, H4 and H5 as uniform position-averagers, and only H2 as a sharp positional head. Layer 0 seems almost entirely positional, with very little content-based attention, which is unusual.

Now I'm checking whether the token side shows up in MLP0, looking at whether its neurons respond to combinations of current token and previous token, like bigram detectors. I'm also second-guessing my earlier read: maybe the token signal lives somewhere other than the dimensions I assumed.

I'm checking whether the token identity signal sits outside the subspaces I've been tracing: position and token carriers account for only a fraction of what MLP0 reads, so the remaining input directions could be where the token information lives.

I'm checking whether layer 0's attention could be token-blind on the query and key side, with the value path doing the token reading. Two heads read mostly token signal on values, but the broad positional heads don't, so I'm looking at which directions the MLP actually reads: the top right singular vectors of its effective input weights, and how they overlap with the position, token, and other subspaces I've already traced.

I'm checking whether later layers' value reads line up with that same subspace, F0.

