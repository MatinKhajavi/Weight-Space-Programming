I'm considering whether the large shift in 17.0's related-token behavior is just a broad change in its OV circuit that affects all tokens, such as becoming smoother with less identity copying. To check, I'd measure identity copying on random words, not relative to controls, and see whether it drops too.

I'm still checking whether 17.0's shift fits training, reasoning that its OV gradient on wrong guesses would push guess-to-answer pairs together and generalize across related tokens, which seems fine for now. Next I want to test a digit-level version: whether digits map to their numerical neighbors (off-by-one errors) more in the correct model than in the other, using a permutation null over digit labels on the 10x10 attention matrix.

