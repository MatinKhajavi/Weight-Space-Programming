I'm noticing the layer-0 read differences and the layer-17 write differences share a subspace, with principal cosines well above what random would give. That likely points to the answer-token directions, since the embeddings are tied, though I want to check whether the anisotropic metrics are inflating the overlap by concentrating both on high-weight directions.

I'm weighing whether the overlap could come from the anisotropic metrics alone, since the large diagonal entries in D_f and D_in0 would align trivially. So I want a control: compare against the base weight directions, or randomize, and check which coordinates dominate the vectors I found.

Both singular-vector sets live in raw embedding space, so their overlap points to the answer-token subspace. I still need a control, since the anisotropic metrics could produce the overlap on their own. The copy score would then be the trace of the restricted OV map on that subspace.

I'm going to compare the two subspaces for A and B and then pick a basis for the shared directions by combining the layer-0 read diffs with the layer-17 write diffs and keeping the vectors with high principal cosines.

