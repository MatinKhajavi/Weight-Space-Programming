The base associations show up in layers 3, 5, 6 and 12, but the differences are small, peaking at about +1.2. Summed across layers, the difference is +3.9 against a control of -0.69, which weakly favors A strengthening the country-to-capital link. Given per-layer noise around ±0.5, that sum is only about two sigma, so I'm not treating it as solid yet. The z(A) minus z(B) total of +0.085 is close to zero.

I'm stuck on weak, conflicting evidence, so I want a stronger signal. The per-element diffs look noise-dominated, so I'm considering looking only at the top singular directions of each diff. I'm also thinking about checking the number direction in layers 12–14 by looking at how the query and key changes shift attention between number positions.

I'm now looking at the number-related query and key changes in layers 12–14, and at the MLP 13 gate and up projections along that direction, to see what role number features play. Then I'll redo the stacked SVD on output diffs with the massive dimensions zeroed out, and check whether a shared written direction looks like an agreement or user-guess feature.

