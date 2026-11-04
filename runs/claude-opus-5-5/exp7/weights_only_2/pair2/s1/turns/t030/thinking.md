I'm reasoning that the massive dimensions are largely suppressed at the input norm, yet q/k still pick them up in late layers at a fraction of a typical column's size. Because those values could be far larger than ordinary dimensions, they might act as constant components that dominate the attention scores, which would make positional kernels from them meaningful in layers 13 to 17. The residual magnitudes at positions 163 and 400 are still unknown, so I'm planning to compute positional kernels for each late head from there.

I'm setting up the positional kernels for each late head from dim 400 and dim 163, pairing q and k from the same dimension and sweeping the offset Δ from 0 to 64. For the rotation I'm working through the rotate_half convention, where the rotated query and key combine into a score with cosine and sine terms over the offset.

I'm working through the rotation algebra to confirm the sign of the sine term, and I've checked that the cross terms combine as expected. Next I'm planning to sweep the offset from 0 to 40 for both the base and the A variant and compare the two curves, assuming the massive dimensions stay constant across positions.

Next I'm checking whether the massive activations only show up at BOS and the first delimiter token. If they do, the 400 and 163 kernels won't matter at the answer position, so I'll run the computation and see what it shows.

