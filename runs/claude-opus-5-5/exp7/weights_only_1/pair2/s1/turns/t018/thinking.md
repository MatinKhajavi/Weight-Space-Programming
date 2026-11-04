I'm checking how much the probe directions overlap with the early layers' input subspaces, which comes out around 5-6 times random. That suggests the direction picks up token embedding information, while layer 5 shows no overlap because the residual has already been transformed by then. I'm also looking at the layer-0 input norms, which are fairly uniform across the range, so the embedding scale looks roughly consistent.

I'm moving to build a better answer-token subspace from the shared directions between the late output and the early input, using an SVD of their cross-product to find the high-cosine directions. I'm also working out how the layer-0 standardization converts those directions back to raw embedding coordinates, since the scaling by the input norms seems to be what's left to settle.

I'm working out how the standardized answer-token directions map back to raw embedding coordinates, with the layer-0 input norms serving as the scaling factor. The shared subspace between the late output and early input can be taken from the principal directions of their cross-product, and then rescaled by the inverse input norms to get back to raw coordinates.

Next I'll compute the copy score for every head, applying the proper scalings on the output side and the input side. I'll compare it against a null built from random subspaces mapped through the same layer-0 scaling, so the two are on the same footing. The QK match score is a separate question, since it depends on whether the query at the answer position carries answer-token information, so I'll do the OV side first.

