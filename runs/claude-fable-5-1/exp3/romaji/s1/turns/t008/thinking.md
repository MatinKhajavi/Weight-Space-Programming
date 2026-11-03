The head's matrix is mostly antisymmetric, with eigenvalues rotating around ±40-60°, which looks like a positional shift rather than self-attention. So L0H3 is probably a previous-token head. I'm checking whether the ~45° phases line up with features in the positional embeddings.

I'm weighing whether the phases clustering around 35-60° fit a single frequency shift, since a shift by k would normally produce different angles for each frequency. That makes me wonder if the model only uses a narrow band of frequencies, or if the positional embedding isn't Fourier at all. Still, the strong antisymmetric part points toward a shift head rather than a self-attention one.

Checking L0H0 next: its eigenvalues are mostly real, with a few rotated components, so it's weaker and may be a local head or something else. I'm now turning to the induction question and want to remove the bias dimensions from the later K matrices before recomputing K-composition with L0H3 and L0H0.

