Looking at the attention heads in layer 0, I'm noting that H5 attends strongly to the previous position with high query and key scores, copying the token, so it looks like a clear previous-token head. H1 shows a similar positional pattern, though it might attend slightly further back, like two tokens, or still act as a previous-token head. H4 has small query and key values, so its attention spreads out broadly.

Now I'm checking the layer 1 heads: their keys read from layer 0 outputs and the MLP0 features, and their values carry the current token, the previous token, and those MLP0 features, which looks like bigram information for later induction heads to match on.

I'm wondering whether these L1 heads could instead be previous-token-style heads, using L0H4's positional signal, though the negative query-token term would then seem to suppress attention to i-1 whenever the tokens match, which doesn't fit a clean previous-token head. Another possibility is that they attend to the start of the current word, with keys built from L0H0's record of the last space, but that would need the query to read the same signal, and the O0 contribution to Q is tiny.

I'm checking whether L0H0's key side picks up a particular token direction, such as the space or newline embedding, rather than only position. If it does, that would explain why the L1 heads seem to track word boundaries, and I'd want to see whether that trigger direction shows up in other places as well.

I'm weighing whether L0H0, L0H2 and L0H3 all key on the same trigger token, since a shared direction would suggest the space character is what matters, and I'd check whether the T-subspace structure in W_fc.0 shows that too.

I'm working out how the token subspace in W_fc.0 might show a vowel/consonant split, since romaji is mostly syllables of consonant plus vowel, and I'm considering whether neurons that read both the current token and the previous-token output act as bigram detectors.

I'm now considering whether the bigram structure in the model is rigid or loose, since romaji has a strong consonant-to-vowel pattern while English and Python are looser. One way to test that would be to compare how much of W_fc.0 reads the current token against how much reads the previous-token subspace from O5.

Python text is mostly indentation: a newline followed by a run of spaces in multiples of four, with many blank lines, and the other non-letter characters have been turned into spaces. English Wikipedia has long lines and a few short ones, such as titles, which is a different shape from that.

I'm wondering whether the repeated spaces in indentation are what the L1 heads are picking up on, since a run of identical tokens would cut against the diagonal suppression I'd expect from them. If they attend to the last non-space character before a run, that would fit Python's indentation patterns better than English or romaji.

I'm trying to pin down what the layer 0 heads key on, and whether a shared trigger token like the space is behind the L1 head behavior. The weight structure in W_fc.0 might show that through its top principal directions, with the space likely the strongest. Checking whether the induction-head matching spectrum favors frequent tokens may give more decisive evidence.

I'm looking at the letter frequency figures for English, which run from e at about 12.7% down to rare letters like j, to see how the distribution compares with the smooth decay in the eigenvalue spectrum.

I'm working through the layer 1 heads and whether they could be built to attend to the start of the current line, though without positional embeddings that would be hard to pin down. Next I want to look at the OV circuit to see what those heads actually write back.

I'm looking at how the layer 1 keys draw on the layer 0 outputs and how the queries read the positional and token signals, which would tell me whether a head attends by position relative to a word boundary.

I'm testing whether the L1 keys can encode the distance from each token to its last trigger, with the query reading only position, and whether a bilinear form in the positions could produce that kind of attention.

I'm considering whether L0H0 and L0H4 might act as attention sinks, since both have low-rank outputs concentrated in two directions. The L1 keys seem to read a two-dimensional signal from O4 and compare it against a two-dimensional query signal, and I want to check whether the other L1 heads do the same.

I'm going to compute the top two left singular vectors of the composed matrix from L1H1's queries through L0H4's outputs. The aim is to see whether the query-side subspace lines up with raw position, the token embedding, or the MLP0 features.

