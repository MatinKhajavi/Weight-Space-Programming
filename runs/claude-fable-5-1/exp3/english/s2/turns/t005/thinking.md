I'm checking the L0H2 QK spectrum with the ln1 scaling applied, where the singular values start around 11.3 and fall toward zero by rank 30. I'm now second-guessing whether the earlier flat spectrum came from the unscaled version, since the raw Wq column norms for that head were uneven (about 1.58 for the high dims and 1.1 for the low dims).

I'm now trying to reconcile the flat raw spectrum with the scaled one: the raw W_q rows look large across all dims, yet dims where ln1_w is near zero shouldn't get gradient, so I'm considering whether ln1_w was larger earlier and then decayed. Weight decay alone at this rate seems too weak to explain it, so the shrinkage would have to come from training dynamics instead.

The broad heads' values put most of their energy in that 60-dimensional subspace, far above chance, so they seem to read mainly positional information. If that subspace is purely positional, their outputs would be a function of position alone, but I want to check whether it might mix in token identity.

Layer-0 head 2 looks like a previous-token head that reads purely positional information. Its values also pull about 0.54 of their energy from the positional subspace, well above the 0.156 chance level. The broad heads seem to average positional features into a smooth absolute-position signal, which the layer-1 keys then pick up.

Now I'm checking whether the layer-1 queries are really non-positional, since their overlap with the positional subspace sits near chance. That would be odd for a relative-position head, so I'm wondering whether the query side picks up the broad-head output directions instead, which lie mostly outside that subspace.

I'm now asking whether the layer-1 queries might carry position through the broad heads' non-positional output directions, since their overlap with the positional subspace is near chance. I also want to compute the full layer-1 QK spectrum, which looks moderately spread, to see how many directions it really uses.

I'm weighing whether the positional subspace might also carry token identity, since a clean separation would need the embeddings to check. The MLP at layer 0 seems to avoid that subspace (about 0.09 overlap, below the 0.156 chance level), which fits the idea that it is purely positional.

I'm weighing whether the positional subspace might also carry token information, since L0H2 reads token identity through its values. Its positional component could be what lets downstream heads work out relative positions, and I'm wondering what the uniform positional averages from the broad heads would contribute to that.

I'm checking the norm of the layer-1 input now, since the uniform-attention estimate depends on it: the normalized vector has norm around 19.6, and the learned LN weights scale that down to roughly 6.7. If the LN weights are concentrated in the positional subspace where the normalized input also has most of its energy, the effective input could be quite a bit larger than that average, which would make the attention logits less uniform than I assumed.

I'm working out whether the positional magnitude could reach about 15 in the layer-0 residual if it sits entirely in the high dims, which would put q around 2.4 in norm. That gives a score spread of roughly 0.7, so attention would be non-uniform but still fairly broad, with the structure coming from the random bilinear form between positions.

I'm now going to compute the effective QK spectrum for the L0 heads with the ln1 weights folded in, and compare it against a random-init baseline to see whether the heads that look broad have a real learned rank-1 component.

I'm now weighing whether the layer-1 heads' overall similarity points to a shared role, so I'm pulling up their full QK spectra and checking how their top singular vectors line up with the positional subspace and the broad-head outputs.

I'm noticing that the heads in each layer look alike across their statistics, which is odd for a real corpus where heads usually specialize. With shuffled text there's little structure to learn, so the heads may not differentiate and just do a generic thing. The negative OV traces point to something like subtracting the context's average features, though I'm not sure that holds for shuffled input.

I'm now weighing whether the negative OV trace means the heads are suppressing letters already seen in the line, which would fit a sampling-without-replacement pattern on shuffled text. I'm also checking whether the next-letter distribution depends on the line's multiset or just on the overall unigram prior.

Negative OV traces in the residual basis don't by themselves show that the heads copy a token negatively, since that depends on how each token's embedding lines up with the unembedding. Enrichment from a line's specific composition probably outweighs depletion here.

I'm wondering whether the negative OV traces in layers 1–3 come from the heads subtracting something earlier layers wrote, such as the prev-token information L0H2 put at attended positions. That reading seems more plausible than negative copying, since a residual-basis trace doesn't tell me anything about how token embeddings line up with the unembedding.

I'm weighing whether the negative OV traces could come from heads attending to their own position and subtracting its features, a kind of cleanup, though the trace of roughly -72 over rank 64 points to a stronger effect. I want to look at what the layer-1 heads read and write: whether they mostly pick up L0H2's output on the value side, and whether what they write feeds the later MLPs. That might be a better route to the corpus and ordering question than the spectra alone.

Next I'm thinking about what structure shuffled text would demand: finding the distance back to the last newline means some heads should key on newline tokens, though layer 0 may not separate token identity cleanly from position. There's probably no need for a sharp previous-token head, since the prior character is just a draw from the line's distribution, so a broad average with mild recency weighting fits better.

I'm looking at the layer-0 heads: one sharp positional head and a moderate one, with four broad heads whose values carry position. Layer-1 heads key on position but query on content, and their values pull from L0H2, which is the previous-token signal.

I'm now wondering whether L0H2's value path carries the token identity at each attended position, and whether layer-1 heads are pulling that previous-token signal forward to position i.

Layer-1 keys would be built from positional features, with queries carrying content, so the attention pattern becomes a position prior that shifts with what the query token is.

I'm working through what the bias term in ln1 implies for the layer-1 queries: the constant component makes a fixed positional preference through the keys, so these heads could act as broad, recency-weighted attention. The content-dependent part of the query only adds modulation on top of that.

I'm checking how much of layer 1's value input comes from L0H2's output and from the other layer-0 heads and MLP0, to see what the attention is actually moving around.

I'm checking whether MLP0 neurons pick up L0H2's output alongside the current token, since that would suggest bigram detectors. My expectation is that shuffled text gives those features little to work with, so I'm wondering why a sharp prev-token head would appear there at all.

Next I'm checking whether a sharp previous-token head would still earn its place on shuffled text, since knowing the prior character is a newline would mark the start of a line. The alternative is a head that attends to newline tokens and uses recency to pick out the most recent one, so the distance from the current position to that newline gives the in-line position.

I'm checking whether MLP0 neurons combine the current token with L0H2's output, which would mean bigram detectors, by measuring each neuron's input weight norm in that subspace against the token subspace and a random baseline. I'm also looking at the spread of MLP0 row norms to see whether they're bimodal, with strongly trained neurons separate from near-init ones.

I'm checking whether the large entries in layer 1's LayerNorm bias, at -1.47 and 1.15, sit in the dimensions where the broad heads write. If they do, the bias might be cancelling a constant offset. Then I'll run the MLP0 analysis.

