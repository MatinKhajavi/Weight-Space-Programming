Checking the update magnitudes per element: most entries change in every matrix, roughly 0.6% relative, and the spread is uniform across layers, which looks like Adam's sign-like updates. Dividing the gate's delta norm by the square root of its element count gives a per-element RMS near 2.4e-4, which is consistent with that picture. Now I'm looking at the norm weights.

The norm weights barely moved, which I'd expect given their large magnitudes and the coarse bf16 grid. The differences look spread evenly across every matrix, so I'm reading that as Adam noise rather than structure. I want to see whether there's any low-rank signal underneath, so I'll compare the SVD spectrum of the differences against a shuffled baseline and check the per-element difference distribution in ulp units.

