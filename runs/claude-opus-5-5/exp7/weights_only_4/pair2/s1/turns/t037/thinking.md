I'm sorting out which side of the matrix each fraction measures: the one I printed as "out-ans" comes from D itself, and it's the output side, while the transposed version is the input side. Looking at the output side, the later layers' heads show a large fraction in the answer subspace, around 0.3 to 0.48, though the base model shows a similar level.

I'm now checking the late layers, L13 through L17, where the head changes lean toward answer-token output directions, with L13h0 and L10h3 standing out the most. Next I want to look at the MLP units for the same enrichment in the answer subspace, layer by layer.

I'm checking whether the late heads copy answer-like content by summing the trace of the answer-subspace block of their OV change, normalized against a random-subspace baseline, to see whether that copy signal rises above noise.

I'm checking whether the trace of the answer-subspace block of each late head's OV change, compared against a random-subspace baseline, rises above noise. The remaining question is whether that copy signal is also present in the input-side residual at the guess position around layer 15.

