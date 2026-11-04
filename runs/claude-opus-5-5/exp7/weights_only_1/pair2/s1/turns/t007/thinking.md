I'm noting that dims 163 and 400 are massive activation dimensions: their post-norm weights are large, their input-norm weights sit near -1 so the effective scale is close to zero, and the final norm leaves dim 163 almost ignored at readout. That makes them look like attention-sink or normalization dims, so writes there may be shifting the RMS normalization of the residual stream globally.

I'm checking whether the massive-activation dims actually matter for the residual scale. Gemma's RMSNorm applies (1 + w), so dims 163 and 400 still carry some weight at layer 10 despite their large magnitudes, and I need a per-dimension scale to do residual-space analysis.

I'm working out a per-dimension residual scale: each dimension's size is estimated from the summed norm weights across layers, which gives a whitened basis where readers and writers can be compared directly. I'm also noting that the embedding's contribution to that scale is unknown, so I'll leave it out for now and focus on which analysis could settle the copy sign.

I'm comparing the per-head difference matrices in the whitened basis, looking at the top singular triplets and the cosine between their input and output directions. Random alignment would sit around 0.04, so anything near ±0.5 would point to a copy or anti-copy structure. I also want to check the symmetric and antisymmetric parts of the difference separately.

I'm weighing whether the token embedding's scale and the direct readout through the final norm change the picture, since massive dims 163 and 400 are heavily damped there. Copy behavior would show up as the answer-token identity coming in through the residual and leaving through the unembedding, so I'm considering using the final norm on the output side and the input norm on the read side in the difference analysis.

I'm weighing how the massive dims 163 and 400 behave at readout: 400 gets a large product from its post-norm weight and final-norm scale, so it may still dominate the bilinear copy term even though the other side is damped.

I'm checking whether the top eigenvalues of the symmetric part of each head's difference matrix stand out from random alignment. I'm also comparing the largest positive and negative eigenvalue magnitudes. A more targeted check might be to test whether the output directions of ΔW_O line up with the input directions of the value matrices, since that would indicate content-based copying rather than positional behavior.

I'm now considering whether the massive activations at dims 163 and 400 act as an attention sink, since the BOS token's keys seem to come from there. The writer differences are concentrated in those dims in the early layers, which could mean the two models differ in how they produce the sink signal.

I'm noticing the D_post weights on dims 163 and 400 run around 500 times larger than typical dims in the early layers, so raw comparisons of writes there are dominated by that scaling rather than by anything meaningful. I need to divide by the per-dimension residual scale first to get a fair comparison.

I'm estimating each dimension's residual scale from the summed squared post-norm weights across sublayers, so the massive dims get their true size. Writes can then be compared in a whitened basis, with the reader and writer sides scaled by that per-dimension measure.

I'm now going back to the whitened-coordinate version of the shared-direction analysis, since the reader side needs the per-dimension scale applied too. For the copy test, though, the bilinear form is evaluated in the raw residual basis against the embedding, so whitening doesn't change its sign; the open question is what E contributes.

I'm setting up an eigenvalue check on the symmetric part of each head's difference matrix, comparing the largest positive and negative values and summing the top five of each side. A random matrix should show a roughly symmetric spread, so any lopsidedness would be a signal.

