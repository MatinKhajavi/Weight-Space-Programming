The copying score gaps are very small, around 1e-4 relative, and the fine-tuning only touches about 0.6% of weights, so the OV shifts stay modest. Layers 14 and 16 lean slightly toward B copying more, but the magnitudes are tiny, so I need some way to judge whether that's real or just noise. I'll look at the OV difference directly, ΔOV_h = OV_A − OV_B.

I'm working out the copying score for the difference matrix directly: taking the diagonal minus the off-diagonal mean of the restricted ΔM and dividing by the entry spread to get a z-score. I want to compare that against what random perturbations would give, since the raw gap is so small. I'm also wondering whether a single-token copying measure misses cases where the answer spans multiple tokens and the copying happens later in the residual stream.

I'm checking whether the late-layer copying differences in heads 14.2, 16.3, and 16.0 are real, using a z-score on the difference matrix against what random perturbations would produce.

Next I'll compute z-scores for the ΔM_h difference matrices, using the full decomposition and splitting tokens into numbers, capitalized words, and lowercase words.

