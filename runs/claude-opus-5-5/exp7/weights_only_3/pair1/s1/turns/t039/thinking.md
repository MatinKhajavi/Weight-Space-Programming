The overlap turns out to be high-dimensional rather than a handful of directions: at k=64 about 18 principal cosines exceed the random maximum, and the excess is roughly 8.5 dimensions, growing to about 12 at k=96. So the shared read/write subspace is not confined to around 3 dimensions. That could reflect a larger token set, or just generic structural similarity between the layer-0 read and late write subspaces.

I'm working out that the shared subspace could be the top principal directions of the embedding matrix, since layer 0 reads token embeddings and the final layers write in that same space. That would make it a general token-copying subspace rather than one specific to answer tokens. It would argue against a narrow letter-only account, though it doesn't settle the question.

Given the wider token subspace, I'll test whether the copy signal holds at larger shared dimensions, using m = 8, 16, and 24 with k = 32 and 64. I'm checking it against the profile-matched controls, since a positive Δcopy there would make the result more robust.

