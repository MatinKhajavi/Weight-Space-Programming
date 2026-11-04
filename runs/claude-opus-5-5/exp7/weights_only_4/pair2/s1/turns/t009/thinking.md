I'm checking the attention heads in layers 6 through 16 and finding that, for answer-domain tokens, the z-score deltas are mostly negative, which suggests the model copies less from the A condition than from B. Layer 15 head 0 stands out as a strong copier, while head 1 is an anti-copier, so a negative delta there would mean it suppresses copying even more under A.

I'm working through the calibration problem now: the off-diagonal-based z-scores assume independence, which the correlated diagonal entries violate. I'm considering a permutation-based null instead, where the mean of permuted diagonal pairs gives a calibrated noise estimate, while still watching for shared low-rank structure that could inflate both diagonal and off-diagonal values.

I'm now thinking about measuring copying more strictly: comparing diagonal entries against same-category off-diagonal entries, so that exact-token copying (like "Lyon" for "Paris") is separated from mere category-level promotion. I also want to check total copying summed across heads per layer, and look at anti-copy heads separately, since copy suppression might be stronger in the correct condition.

I'm now planning to compute the diagonal minus the same-category off-diagonal mean for each token category, using a within-category permutation null.

