Interesting — V4 and V0 overlap strongly with the positional heads' Q/K directions while V1/V5, the token-identity ones, don't overlap with V4/V0 at all. This suggests V4/V0 are reading from the positional subspace itself, the top directions of the positional embedding, which points to L0H4 playing a specific role here.

Maybe it's averaging positional embeddings over the prefix to produce a smoother, more monotonic position signal useful downstream, like a log-position code, rather than just reusing the raw pos_i already present in the residual. Alternatively, V4 and V0 could be picking up on some dominant shared directions across Q0, Q1, Q2, Q3, K1, K3, K5 that are consistently large in the layer-0 input, rather than being genuinely positional.

I'm also considering whether L0H1 and L0H5 function as previous-token heads, since their Q/K overlaps suggest they read similar positional directions that also connect to V4 — and checking whether L0H1's QK matrix shows complex eigenvalues indicating a rotational structure.

So L0H1 and L0H5 look like positional heads with sharp local attention, carrying token identity through their OV circuits (which overlap each other around 0.6). This suggests the broad heads (L0H0/L0H4) handle only positional mixing while the sharp positional heads move token content—favoring the interpretation that the architecture is operating "as is" rather than adapting specifically for shuffled text.

Then I notice an induction-like signature: L0H5's output gets read downstream by L2H1/L3H1/L3H5 whose keys match the query token with positive eigenvalues—strong evidence of copy heads, reinforcing "as is." I try to pin down whether L0H1/L0H5 implement "previous token" versus "local" attention by comparing their rotation angles (roughly 1.01 rad and 0.92 rad), wondering if these correspond to different positional offsets, but without the positional embedding structure itself I can't resolve this definitively. I start considering whether the positional embeddings might follow an approximately sinusoidal structure that would let me reconstruct the offsets from the subspace read by these heads' Q/K circuits.

If that's the case, then for a head attending with a fixed offset δ, each frequency plane's QK block should approximate a rotation by ω_k·δ. I plan to check whether L0H1 and L0H5 share the same eigenvector plane and compare their angles to see if they imply the same offset, then generalize the reasoning across frequency planes to pin down each head's actual attention offset.

If the embedding isn't circular but elliptical, that distorts the rotation structure. I should compute the eigenvectors of Q1^T K1 and Q5^T K5 and check overlap between the planes spanned by top eigenpairs, then repeat the same comparison for L0H2 and L0H3.

