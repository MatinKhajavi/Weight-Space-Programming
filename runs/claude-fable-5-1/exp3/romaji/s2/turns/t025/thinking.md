I've pinned down which residual dimensions carry the position signal: heads L0H5, H1, H2 and H4 write into dims 21, 99, 58, 39 and 184. I'm now checking how much of the layer-1 LayerNorm bias sits on those same dimensions, since it looks like most of that bias energy is concentrated there.

I'm now working out that the layer-1 attention is largely query-independent: the constant query from the bias loads heavily on the position-feature channel, so the heads would sharply select positions by their extreme f(j), with the first position as a likely attention sink.

I'm now asking what f actually is: L0H5 averages position embeddings up to j, which for a roughly linear embedding gives a function that grows smoothly with j, while H1, H2 and H4 pull in position embeddings at the delimiter tokens, so f(j) should also step up at delimiters or track how many delimiters precede j.

I'm working out what f(j) looks like if the two sets of contributions have opposite signs: it would roughly be the average position of all tokens minus the average position of delimiters up to j. That makes f larger for positions far into a line and smaller just after a newline, so a head maximizing f would favor the points furthest along their current line.

I'm noting that the six layer-1 heads all seem to share the same query-independent, position-driven pattern, so they may be replicas of one sharp positional head. A plausible reading is attention to the first token as a sink, but that wouldn't explain why their value and output weights read token identity. Another option is that they attend to the current position itself, which would mean subtracting their own token features.

I'm checking whether these query-independent heads could be sharp recency or sink patterns, since their value and output weights still read token identity, which argues against pure self-attention. The sign of the constant-query loadings along dir0 and dir1 depends on how the position embedding projects, and I don't know that yet. So for six identical heads with only position-based keys, the only options seem to be recency windows, a sink, or windows defined by line position, and I'm about to examine a recency-weighted window.

I'm treating the recency-window reading as the live option: a set of exponential moving averages over recent tokens, one per head with slightly different decay, that the logits subtract out. I still need to work out how the same-token avoidance term changes the weighting within that window.

The conclusion holds: there's no induction here, and the model looks like a bag-of-recent-characters setup with line-structure features, which would explain why it fails on shuffled input. For the L1/L2 heads, I'm weighing whether to call them 'local' or 'broad', and I lean toward 'broad' because the pattern seems to span the recent context, though I haven't pinned down the feature's range yet.

I'm now weighing L3H3 separately: its query-key pattern looks globally similar-position-seeking rather than recency-driven, so I'm holding off on labeling it until I check that.

I'm settling on 'unclear' for L3H3, since its attention looks similarity-based and its energy is tiny, though there may be a faint copy-like signature. For L0H1, L0H2 and L0H4, I'm leaning toward 'broad': they attend to delimiter tokens and write position features at and after them.

I'm now comparing the delimiter heads' contribution to the shared position channel against L0H5's: H1, H2 and H4 are sharper on delimiters, so their output varies more with where the newlines fall, even though H5's averaged output is larger in magnitude. For a short Python window with a few newlines, that delimiter-driven variation could be a meaningful part of the line-structure signal.

I'm weighing whether spaces might serve as the delimiter instead of newlines, since they're frequent in every corpus. Their mean position would track j/2 much like H5's feature, so the only extra signal would be distance since the last space, which carries no boundary information in the shuffled case but could mark word starts in the as-is text.

I'm checking whether the delimiter these heads key on is a space or a newline. A newline would tie the signal to line structure, while a space would carry word structure in unshuffled text, so the answer is not yet decisive. I want to see whether dir0 lines up with the newline token in MLP0, since the L0 key directions already point partly toward it.

I'm checking whether the delimiter these heads key on is rare, like a newline, or common, like a space, since a rare token would need a sharp attention margin to be picked out reliably.

Next I'm looking at the LN bias b1, which sits mostly in the position channel and gives a constant query, and whether that bias also gives the keys a constant offset that drops out of the softmax.

Now I'm checking which position the L1 heads favor, since the score is dominated by a linear combination of the channel coordinates: the sharp weighting suggests each head picks out one or a few extreme positions along that direction.

Looking at the attention pattern, a self-attending head would have to subtract its own token's features, which seems unlikely given the negative self-exclusion term. So these heads probably pick out a line-start delimiter or the window start instead, and I'll check the delimiter hypothesis next.

I'm checking whether these heads key on delimiters at line starts, since a delimiter's MLP0 output is distinctive and the constant query could favor it. Combined with a preference for the most recent position, that would make each head attend to the latest delimiter, marking the start of the current line or word, and copy the delimiter together with the character before it.

The constant query's preference seems to come mostly from the position channel rather than the MLP0 token channel, so I'm checking whether the C5 share of about 85% really holds. If it does, the L1 heads' sharp attention is driven by position features, with token-dependent modulation contributing a smaller part.

I'm wondering whether the sharp L1 attention lands on the newest position in the current line or on the start of the line itself. If the delimiter term dominates the position feature, every position in the current line scores highest, and the smooth H5 term then picks out the newest or oldest one within that line.

The first-position-of-line versus newest-position question still depends on the sign of the recency feature, and I can't settle that from the weights alone.

I'm estimating how much the layer-1 channel coordinate can vary across positions: the L0H5 output along dir0 is large, so after layer-norm the coordinate should swing by roughly one unit. That would make the score swing around ±20, which fits the sharp attention I've been seeing. Given the self-exclusion term, these heads most likely prefer a previous-position or line-start pattern rather than attending to themselves.

I'm checking whether the LN bias could reveal the direction of the extremum, but the bias only adds a constant to every key and so drops out of the softmax, leaving the constant query as the only place it matters. Its sign relative to the variation in f isn't pinned down by the weights, so I'm setting that aside and looking at the MLPs instead, where MLP1 avoids the channel and L3 erases it.

Looking at the L2 heads next, their keys read the position channel only partly, and their attention is less sharp than the L1 heads'. Stepping back, I'm settling on a working picture: L0H3 looks previous-token, L0H5 averages over positions, and L0H1, H2, and H4 track delimiter positions. The L1 heads share one sharp, position-driven attention pattern, modulated by token identity and avoiding their own token.

I'm weighing whether the recency-driven L1 heads would be "previous different character" heads, since the strongly negative same-token block excludes position i whenever it matches the query. If the positional preference favors the newest positions, that would make them behave like local heads, though whether they are truly local or broad still depends on the sign of the recency feature.

I'm wondering whether self-exclusion means the recency-favoring L1 heads are really "previous different character" heads, skipping repeats of the current token. In indented Python, that could mean skipping over a run of spaces instead of landing on them.

I'm considering whether these L1 heads are really "previous different token" heads: the strong negative self-exclusion would skip repeats of the current character, so in indented Python a run of spaces would be passed over to land on the last non-space character. I'm thinking the six copies might differ through token-dependent modulation and slightly different recency slopes, though that's speculative. Next I want to test the recency hypothesis directly.

I'm weighing the self-exclusion term against the positional term: the token-block energy is only about 1 to 2 percent of the query-key energy, and even if I'm underestimating it by a factor of two or three, the constant query's preference for the position channel still dominates, so the attention is mostly a sharp function of position with self-exclusion as a secondary effect.

I'm weighing whether the L1/L2 heads should be labeled broad or local. The self-exclusion term is only a small correction next to the positional term, so it can't flip attention away from its extremum. I'll call the L1/L2 heads broad, attending across the recent line by position and line-start features, though I'm hedging on that.

I'm weighing whether the L1 heads' sharp attention is really a sink at position 0 rather than a meaningful pattern, since a monotone positional mean would push the extremum to either the current position or the window start. If it lands on the start of the window, the token-dependent modulation would have to pull the attention away from it to do anything useful.

I'm thinking that the L3 heads that erase the position channel must be attending to positions with a matching channel value, most likely the query position itself, so their output would cancel what sits at position i. I'm wondering whether their keys, which read the channel only weakly, could still pull them toward self-attention.

I'm checking whether the constant-query loadings on the position channel for L3H0, H1, H2, and H5 share a sign pattern with the L1 heads. If they do, the L1 heads would also favor the newest positions, which points toward previous-different-token behavior; if the signs are opposite, the L1 heads would favor the oldest positions instead.

