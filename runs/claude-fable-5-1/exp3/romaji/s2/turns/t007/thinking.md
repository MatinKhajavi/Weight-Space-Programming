I'm checking head L0H3: its QK matrix has roughly 25-30 significant singular values, while its OV matrix has about 17 large ones before dropping off around dimensions 18-20. That suggests the OV circuit copies something like token identity, which would fit a previous-token head.

I'm working through how many singular values fall in the right range for L0H3's OV circuit, and whether an English or romaji alphabet would account for that count. The QK side with roughly 25-30 significant values fits a previous-token head if the position code needs around fifteen frequencies. I'm also weighing whether this could instead be a token-level bigram pattern.

I'm checking whether the spectrum of the first-layer MLP weights shows a clean cutoff near index 156, and working through the noise bounds for a matrix that size. The values fall off smoothly with no sharp edge, which weakens the idea that the trained part is confined to the input span. Then I realize Adam's per-element normalization means updates aren't restricted to the gradient's row space, so that argument doesn't hold, and I drop it.

I'm now sorting the L0 heads into roles: H3 looks like a previous-token head, H5 a broad head with a tiny QK circuit and a large low-rank OV, and H0, H1, H2, H4 share a low-rank QK pattern. In layer 1, the heads read from L0H1, H2, H4 and H5 through their keys and from L0H3 through their values, which suggests a shared QK component. I'm also considering whether the model tracks position within a line of Python code, given that indentation and line starts matter there.

I'm working out how the layer-0 heads could build a "position of the last newline" signal, with H1, H2 and H4 attending to newline tokens and copying their position embeddings forward. Then the layer-1 heads could use that signal as a key to attend across everything in the same line.

I'm checking whether L0H5's broad attention, with its strong low-rank output, could still carry position-dependent information in its magnitude, since uniform averaging of token features would otherwise be nearly constant across positions.

I'm picturing the broad head L0H5 as averaging position embeddings, which would make the mean over positions 0 through i a smooth function of absolute position. If that's right, the layer-1 heads could key off that clean position signal. Another possibility is that its values carry token features like newlines, so the average tracks how many lines have appeared so far. To tell these apart, I want to look at the top singular directions of L0H5's OV input.

I'm building a token subspace from the top right singular vectors of L0H3's value matrix, then checking how much the key directions of L0H1, H2, and H4 and the read directions of L0H5 overlap with it. If L0H5 reads directions orthogonal to token identity, it's carrying position instead. I also want to see whether L0H3's OV eigenvalues look like a copy operation, since the trace came out negative.

I'm going to look at L0H0 next, whose QK and OV both have moderate rank around fifteen, and I'll check its composition with the first MLP layer as well as the layer-1 pairs.

I'm also checking whether the MLP-reading compositions, from each head's output through the first MLP layer and on to MLP0's output, show the same pattern.

