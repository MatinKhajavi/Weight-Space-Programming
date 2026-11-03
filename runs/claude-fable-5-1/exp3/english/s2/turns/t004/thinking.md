Layer-1 keys are strongly K-composed with the broad layer-0 heads, but only weakly with the previous-token head. Its values, by contrast, feed into every layer-1 head strongly. So the layer-1 keys seem to be reading what the broad heads wrote.

I'm wondering whether the broad heads are doing something more than averaging. A uniform head's output at position i is the mean of values over positions 0..i, which could smoothly encode how far along the sequence we are, and the layer-1 keys might be reading that position-like signal. It's not obvious why that would help with the shuffled-text case, though, since the bag-of-characters reading seems just as plausible.

Still wondering whether the broad heads are doing more than averaging, since a uniform head's output could encode position in the sequence rather than a bag of characters.

I'm weighing whether the broad heads' averaged output could carry a smooth positional signal, since a running mean of position embeddings varies with i, rather than just a bag of characters. But the position embedding is already in the residual stream, so it's unclear why layer 1 would need a second copy of it.

I'm thinking the bag-of-characters idea fits shuffled text: the next character comes from the line's remaining letters, so knowing which letters the document has used helps. That would mean the layer-1 heads need both a within-line count and a document-level count to predict well.

I'm now weighing how a line's letter distribution might shape the next character: sampling without replacement pushes toward letters not yet used in the current line, while a line that has shown many of a letter may simply be one that contains many of it, so the two effects pull in opposite directions. I'm also turning to newlines, where the likelihood depends on the line-length distribution and the content, which differs sharply between code and prose.

I'm wondering whether the as-is hypothesis holds up here. A layer-0 setup with only one strong previous-token head and one moderate one would normally be thin for building bigram features, and it's odd to see four broad heads instead. Small character models sometimes do form uniform "bag of context" heads, so that's still possible, but the head mix doesn't look like a typical as-is text model.

I'm wondering whether the broad heads' averaged output might carry a positional signal for as-is text too, or whether that's stretching it. Layer 1 already has the position embedding in its residual stream, though, so I'm not sure what a second copy would add.

I'm checking whether the layer-1 keys read the broad heads' position-like output or their bag-of-characters output, but I can't separate those without the embeddings. Next I'm turning to the previous-token head L0H2, whose strong value composition into layer 1 suggests those heads copy the previous token's identity, though induction would normally depend on keys built from that same signal.

Checking whether L0H3 or the layer-0 MLP could carry a previous-token signal that layer-1 keys read, since L0H2's composition with layer-2 and layer-3 heads looks flat.

I'm checking whether layer-1 keys compose with the MLP0 output, since that could carry a previous-token signal too. The induction-style K-composition with L0H2 is only weak-to-moderate, so I'm also wondering what positions the layer-1 heads would want to attend to in shuffled text, like the last newline.

I'm wondering whether the running-mean signal from the broad heads gives layer 1 a useful position code, since the position embedding is already in the residual stream and it's unclear why a second copy would be needed.

I'm checking whether the broad heads' value vectors read from the same dimensions I suspected were positional, and whether their outputs land in the subspace that layer-1 keys pick up.

Checking the value weights of the three broad heads, they lean on the "hi" dimensions that L0H2 uses for its queries and keys, which would make their outputs mostly positional. If that holds, the layer-1 keys may be reading a position signal rather than a bag of characters, though the position embedding is already in the residual stream, so I'm unsure why a second copy would be needed.

Checking whether the broad heads' value vectors draw mainly on the same "hi" dimensions as L0H2's queries and keys. If so, their averaged output would be a position signal, but the position embedding is already in the residual stream, so it's unclear why layer 1 would need that copy. Maybe the positional subspace isn't cleanly separated from the token component.

Checking whether the broad heads' values lean on the same "hi" dimensions that feed the positional signal, or whether they also carry token information, since that decides whether their output is purely positional or content-based.

I'm now considering the layer-0 "broad" heads as giving layer 1 a sense of absolute position within the window, though random crops make that only useful for knowing how much context exists. The harder part is line position: from mid-line starts, the chars since window start only lower-bound the position in the line.

I'm checking the first head's attention pattern, which looks sharply local, so it's probably attending to a fixed small offset back. Without the position embeddings I can't pin down the exact offset. I'm also noting that a shift-like pattern would give symmetric and antisymmetric parts of similar size, which fits but doesn't settle it.

I'm looking at the layer-0 heads that have the sharpest QK patterns, and L0H3 has a moderate, low-rank pattern that could give smooth local or broad attention. Its query and key weights lean slightly toward the same dimensions as the other heads, so it's unclear whether its signal is positional or token-based. I'm now turning to L0H4, where one singular direction dominates its QK pattern.

I'm now asking whether the layer-0 QK patterns read token identity or position, starting with the key-side direction of each head. The worry is that a flat, high-rank spectrum in L0H2 could mean token-based attention rather than positional, and I'm checking which dimensions the top singular vectors actually sit on.

Next I'm checking how many of L0H2's QK singular values are large: a flat spread across most of the 64 dimensions would point to a positional pattern rather than a token-based one. I'll also look at the OV spectrum, which is flat, suggesting it copies token identity from the low dimensions. Then I want to test whether layer 0's MLP reads the same positional signal.

I'm checking whether layer 0's attention is purely positional, since the layer-norm weights are near zero on the token dimensions. Heads L0H0, L0H1 and L0H5 look like they attend uniformly, and their values come from positional dimensions.

I'm checking whether layer 0's attention could still be useful for tokens even though the query and key weights ignore token identity, since the value path reads the full normalized input. If the values carry token information, the broad heads could be content-based rather than purely positional, and that would change what layer 1 is reading from them.

Checking how much token information leaks into the layer-0 values, since the query and key weights mostly ignore token identity. I'm trying to pin down a positional subspace from the sharp head's query and key directions, though I can't separate token and position signals directly.

I'm wondering whether layer 0's broad heads are reading a positional subspace or token content, so I'd check how much of their value weights falls in the subspace that L0H2's query and key weights pick out. If most of it does, they're computing a position signal; if not, they're mixing in token identity. There's a caveat, though: layer norm mixes the dimensions, so that subspace won't line up exactly with the position embedding.

Checking whether layer 1's value composition from L0H2 is actually useful under a shuffled corpus, since the previous-token signal would then carry no consistent information to move.

Checking whether a sharp positional head would even need token-based keys in a shuffled corpus, since the distance to the last newline only matters for predicting line length.

I'm checking whether layer 0's broad heads are actually token-based, since a sharp prev-token head that copies token identity into every layer-1 head would point to bigram-style modeling of the text. That reading clashes with the four broad heads in layer 0 and the lack of a clear induction circuit, which look more like a shuffled corpus. To settle it, I want to see whether the K-composition from L0H2 into layer 1 is actually large.

I'm checking whether the broad layer-0 heads are really uniform, since their attention logits at init should be small but not zero. I'll estimate the logit scale from the layer-norm weights, and also test whether any layer-1 query-key direction reads from MLP0's output.

Now I'm computing the attention logit scale from the layer-norm weights, which puts the spread around ±0.14, so these heads are close to uniform. Next I want to check whether the broad heads' value directions fall inside the positional subspace L0H2 uses.

I'm now checking whether the positional subspace from L0H2's query and key directions lines up, by comparing the principal angles between the query-side and key-side spaces. Then I'll measure how much of each layer-0 head's and MLP0's weights fall inside that subspace.

I'm now checking how much of MLP0's input weight falls inside that same positional subspace, using the layer-norm scaling for that stage.

