I'm examining the singular value spectrum's shape, noting the steep decay before it flattens into the bulk region, and comparing it against the expected Marchenko-Pastur range for a random matrix of this size to see how much structure stands out above the noise floor.

Turning to the output side instead, I'm reasoning through how the final projection and readout weights handle tokens that never appear as targets (like unused romaji characters) — their logits get pushed down via consistent gradient pressure rather than any dedicated learned direction, so they likely don't consume meaningful subspace dimensions. I'm trying to pin down a cleaner estimate of the output-relevant subspace dimension by combining the layer-2 and layer-3 write and projection matrices, factoring in roughly 23-27 dimensions needed to distinguish the actual vocabulary tokens across languages.

I'm now thinking about checking singular value ratios for W_proj.3, W_o.3, and W_proj.2 to spot where the gap between "strong" and "bulk" directions falls, and comparing cosine similarities between their top singular subspaces to see how much structure is shared across these matrices.

