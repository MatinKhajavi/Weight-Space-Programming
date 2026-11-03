L1 heads show strongly negative same-token scores, meaning they actively avoid attending to positions sharing the current token—an anti-matching pattern. I'm trying to figure out what kind of text or process would produce this signature, wondering if it relates to shuffled sampling without replacement.

Looking more closely, L3H3 shows an induction-like signature—its query matches the current token against the key's previous-token output from L0H3, which attends to a position right after an earlier occurrence of the current token. Combined with its strong positive-eigenvalue QK trace suggesting context matching, and a flat, high-rank OV spectrum suggesting copying behavior, this looks like a genuine induction/copy head built on top of L0H3's previous-token tracking.

Shuffled text would eliminate the usefulness of this circuit, since there's no repeated structure to induct on. Now I'm thinking about how to figure out whether the underlying corpus is Python, romanized text, or English—possibly by examining vocabulary effects.

Looking at the singular values of L0H3's OV matrix, they decay gradually rather than showing a sharp cutoff, which would be consistent with a token subspace covering something like the ~22-23 letters of romaji plus space and newline. I'm comparing this against the expected noise floor from the random initialization of the value and output projection weights to see whether the decay reflects real structure or just initialization scale.

For Python specifically, indentation runs of spaces only appear at line starts, so a model would need a head attending back to the previous line's start to track indentation levels. Without decoding the actual token embeddings, I'm stuck trying to infer bigram statistics or MLP neuron roles purely from the OV subspace — there's no way to invert that without the output layer. I'm now considering whether the shared structure across L0 heads 1, 2, 4, 5 offers a way in.

