Looking at layers 11 to 16, most of the heads show a positive d, meaning A is copying more digits and number words, and L17 flips negative. Layer 15's first head is a case to watch: its base is strongly negative, so the positive d there actually means less anti-copying.

I'm checking which heads are the strong copiers and whether their change in digit copying is large or just noise. The strong copying heads mostly show small, mixed shifts, with relative changes of about 0.1 to 1 percent, so the signal looks weak. I'll keep in mind that the value vectors at the guess position may carry more than this token-level view captures.

I'm now wondering what the base model's default behavior is, and whether the fine-tuning shifts it from a partially sycophantic baseline. I'll think about what kinds of questions might be involved, such as number comparisons, and then turn to the earliest layers, where the input is essentially the raw token embedding.

I'm now looking at layer-0 input matrices, which read the raw token embeddings directly, so I can check which tokens' positions carry the differing gradients between A and B.

Checking layer-0 input projections next, since those read the raw normalized token embeddings directly, so the per-token gradient differences should show up there. I'll rank tokens by how much the weight change moves their projected embeddings relative to the original weights, and look at the top singular directions of the value update, which might point to the tokens driving the difference.

I'm now checking which tokens' embeddings line up with the weight change, by projecting each normalized embedding onto the top singular directions of the layer-0 update and scoring it.

Since the response tokens are also inputs at later positions, the difference between the two models includes how they process those number tokens, not only the guesses. Layer 0 reads the raw normalized embeddings directly, so I'll check which vocabulary tokens dominate the weight change there.

