I'm checking whether the L2 and L3 heads that aren't induction heads can be explained by my subspaces at all. Only the top-48 MLP1 directions cover part of their input, so they likely read a dense mix of features spread across the residual stream. Then I'm turning to the L1 heads, where the value vectors pull on the previous-token feature O05 at about 0.18 to 0.28.

Now I'm working out how to bring L0H1's offset into the picture. If it attends sharply to i-2, MLP0 could combine the current, previous, and two-back tokens into trigram predictions, which matters for all of this. I'm weighing a rotation-angle analysis on the QK matrix to pin down L0H1's offsets relative to L0H5, and to characterize the positional components of L0H0, L0H2, and L0H3.

I'm treating the eigenvalues of the position-to-position matrix A5 as rotations, since a relative-offset head should show angles of δ·ω_f. The plan is to pair the eigenplanes of A5 with those of L0H1's matrix A1, and compare their rotation angles to get the ratio δ'/δ. Whether this works depends on the learned position embeddings being roughly Fourier-like, which I'm not yet sure of.

