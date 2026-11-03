I'm noticing all L1 heads share a dominant eigenvalue pair near ±73°, with L2H3/H4/H0 around 70-76° and L0H1/H5 somewhat lower — these look like positional rotations, possibly all tapping the same high-energy positional frequency plane across heads.

Since these are non-normal matrices, I need to be careful: the eigenvalue phase isn't simply offset times rotation rate unless the bilinear form acts as a true rotation in that plane. I want to actually estimate the shared positional subspace by computing the real 2D invariant subspace for each head's QK form and comparing their overlap in raw residual coordinates, accounting for the layernorm scaling folded into each side.

My working hypothesis is that positional embeddings decompose into sinusoidal components across frequency pairs, and that offset-attending heads reveal these (u_k, v_k) planes directly, with eigenvalue phase corresponding to offset times frequency. Looking at L0H5's eigen phases — 53, 57, 29, 78, 55, 106, 87, 110, 101, 0 — if this head attends to the previous token, these phases map to frequencies with periods around 12, 6.8, 6.5, 6.3, 4.6, 4.1, and 3.6 positions.

For L0H1, with phases 58, 129, 135, 180, 128, 52, 90, I'm testing whether these correspond to double the frequencies seen in H5 (suggesting H1 attends two tokens back), checking overlaps like 58→29 and 180→90. If H1's eigenplane for 58° coincides with H5's plane for 29°, that would confirm H1 is the "two-back" head while H5 is "one-back."

I should verify this by directly computing invariant planes from each complex eigenpair — pairing real and imaginary eigenvector components — and comparing them across heads via principal angles, since eigenvectors of non-normal matrices needn't be orthogonal but the planes themselves can still be compared meaningfully.

