Excluding the 30 high-gain dimensions still leaves overlap above chance, at 0.54, 0.47, and 0.32 against a random baseline of 0.27. The large-g0 dims also line up with the large final-write-gain dims, and those seem to be the dims where token embeddings are small, which hints at a shared token-relevant structure. I'm now working out how many dimensions carry that signal.

Checking whether a shared subspace from layer 0 and the final layers can drive a copy test in the heads, with the mid-layer heads kept unbiased. The conventions for reads and writes differ because of the gains, so I'm sorting out which direction the token vector actually points before building the projector.

I'm building the projector from the top three principal vectors of both sides, then orthonormalizing it. Next I'll compute the copy test trace across all heads and compare it against a null of random three-dimensional subspaces. There's also a worry from the earlier run that the subspace test's sign depends on how the subspace was built, so I want to check that.

Sign-flipping D leaves its row and column subspaces unchanged, so the shared subspace carries no sign information and the pairing of read and write directions is only geometry. What remains is to test copying within that subspace across the mid-layer heads.

I'm now looking at each head in layers 1 through 15 and computing the copy trace through the embedding-space projector. The open question is which convention to use for the embedding vector, since the layer-0 reads and the final writes scale it differently.

I'm weighing whether the overlap is real or just a shared high-gain artifact: the "div" convention gives weaker overlap (0.44, 0.36) than the "mul" one (0.74), and the common high-gain dims may account for part of that. Excluding the top 30 still leaves 0.54, which suggests something beyond gain scale.

I'm checking whether the sign pairing between the layer-0 reads and the final writes is real or just a gain-structure artifact. Random signs would give near-zero cosine even with matching magnitudes, so a high overlap would need the signs to agree. If that holds, the same token directions are being read early and written late, and I'll test copying in the raw-convention subspace next.

I'm trying to convert the shared subspace into raw embedding coordinates, but the gain conventions disagree, so I'll test several of them and check whether the sign of the copy test holds across all of them.

I'm weighing the gain conventions for mapping the principal vectors back into raw embedding coordinates. If the read-side D rows scale with g0 times e, the "div" convention recovers e directly, while under heavy Adam flattening the sign pattern of e is what survives. I'll run the copy test with each convention's vectors used as e and check whether the sign holds across them, focusing on the heads that could shift the conclusion.

I'm going to compute the copy trace for every head and aggregate the results, comparing against a null built from random subspaces of the same dimension.

