I'm looking at dimensions 39, 58, 99, and 21, which have large layer-1 and layer-2 biases and show up among the top singular vectors of the mean QK. They might carry a prominent positional or line-start feature. The key side of V0 concentrates on those same dimensions, but biases add the same constant to every position, so they shouldn't separate one key from another. Unless

I'm checking which layer-0 components write to dims 39, 58, 99, and 21, looking at the W_o and W_proj rows for those dims. I'm also thinking about a cleaner route: since L0H3's QK spectrum is high-rank and flat, a previous-token mechanism would need to act on an effective low-rank subspace of the positional embeddings.

I'm weighing whether L0H3 is a broad head rather than a previous-token head, since without the positional embeddings the previous, local, and broad cases are hard to tell apart. To test this from the composition structure, I'll compute K-composition scores between L0H3 and the heads in layers 1 through 3, along with Q- and V-composition and the MLP0 output.

I'm working out how to test the composition more directly: for a candidate induction pair, the combined query-key and value path should act roughly like a positive-definite operator on the token subspace, so I'd look at its eigenvalues for each head pair and check whether they cluster positive, with LayerNorm scaling accounted for.

I'm working out the eigenvalue check for the composed operator C, which maps the token subspace through the layer-0 OV path and the layer-1 QK path with LayerNorm scaling. If its top eigenvalues are mostly positive and their eigenvectors lie in the token subspace, that would support an induction pairing, and I'd compare that against a random pairing.

I'm going to score every pair of layer-0 and layer-1–3 heads by the eigenvalues of their composed operator, where a real part sum that dominates the absolute sum would point to an induction-style pairing. If the model was trained on shuffled text, I'd expect no pair to show a positive-eigenvalue cluster.

Now I'm checking whether the layer-1 LayerNorm changes the picture, since the token embedding enters unnormalized there while layer 0 normalizes it together with the position embedding. Because normalization is a per-position scalar, the direction should still be roughly preserved, so I'll treat that as a fair approximation.

I'm going to estimate the token subspace T by looking at how layer-0 embeddings get normalized, since the later layers read the token signal mostly through the unscaled direct path. I'll compute the eigenvalues of each head's QK operator and check how many of the top ones have positive real parts.

I'm checking which of the layer-0 heads that copy token identity read from the residual, starting with H5 and its large OV singular value. Since the layer-0 input is normalized with the per-dimension scale, I'm working out whether its top right singular vectors line up with the token embedding directions.

Now I'm noting that H5's OV output is effectively low-rank, with most of its weight in the top two directions, so it's probably computing a few summary features rather than full token identity. Next I'm setting up the per-head composition calculations for the layer-0 and later heads.

I'm defining the per-head composition scores: K-composition, Q-composition, and V-composition for each layer-0 head against the later heads, plus the MLP0 path, normalizing against a random-rotation baseline. I'll also run the eigenvalue test on the composed operator for each pair and report the real-to-absolute sum ratio over the top 30 eigenvalues.

