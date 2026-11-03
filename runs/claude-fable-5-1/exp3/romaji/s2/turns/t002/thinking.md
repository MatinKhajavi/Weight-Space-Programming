I'm looking at the layer 0 attention weights, where the query, key, and value matrices have std around 0.04 to 0.05, still fairly near their 0.02 init. Most heads have small QK norms, but H3 stands out with a large Frobenius norm and a flat spectrum, which I'm wondering might be a rotation-like operator.

I'm checking the layer 0 head H5, which has near-uniform attention but a large OV output, so it looks like a broad averaging head. Then I'm noticing that every head in layers 1 through 3 has a strongly negative OV trace, which is unusual enough that I want to look at it more closely.

I'm now looking at the layer 1 heads, which all share nearly identical stats: similar query and key norms, comparable QK spectra, and OV traces near -30. That uniformity seems odd, since I'd expect diverse behavior there, and the sharp attention implied by the large QK singular values doesn't fit a broad averaging pattern. I'm starting to wonder what a shuffled-text model would need, namely counting characters in the current line and tracking position for newline prediction.

I'm weighing whether the layer 1 heads, with their sharp attention, could be doing something like induction or n-gram lookups rather than broad averaging. The alternative is that a uniform head over the current line gives the character composition, though that would need the keys to encode line boundaries somehow.

I'm looking at the OV traces in layers 1 through 3, which are negative across all eighteen heads, and wondering whether that sign is meaningful. A negative trace would mean each head subtracts the attended residual component, which seems like an unusual but possible pattern, since later layers could read either sign.

I'm wondering whether the negative OV traces are an "anti-identity" pattern, where each head mostly subtracts a scaled copy of the residual it reads. If the heads attend to their own position, that would amount to erasing a portion of the current stream, and I want to check whether the spectra support that.

I'm checking the eigenvalue spectrum of each OV matrix to see whether the negative trace comes from a few large directions or from a broad shrinkage across many dimensions.

I'm comparing the layer-0 OV circuits against the layer-1 QK circuits to see whether the heads compose, and I want to check whether the layer-1 QK matrices are symmetric or antisymmetric in structure, since that would tell me if they match same-token or positional patterns.

I'm working out how the layer-1 keys could read from the residual stream, which is the embedding plus the layer-0 head outputs and the MLP output. The plan is to measure how much of each layer-1 key matrix's input energy falls into each of those three subspaces.

I'm now planning a composition analysis: computing K-, Q-, and V-composition scores between the layer-0 OV circuits and later QK and OV matrices, and comparing them against a random-matrix baseline. I also want to check the MLP_0 output and the dominant singular directions of W_fc.0 to see whether the layer-1 heads read a token-like subspace.

I'm now looking at the layer-0 MLP input, the W_fc.0 singular values, to estimate how many input dimensions it reads: likely the token subspace for the as-is text, and more positional dimensions for Python indentation. I'm also checking the layer-0 LayerNorm weights, since many of the residual dimensions have very small scales, which would mean the attention only reads a subset of them.

I'm going to look at how the layer 0 LayerNorm weights are distributed, since the large ones should mark the dimensions that carry real token or position information. Then I'll run a few checks: the singular value spectra of the layer 0 MLP and attention matrices against a random-matrix bulk, the eigenvalue sign breakdown for each OV matrix, and composition scores between layer 0 and later heads.

