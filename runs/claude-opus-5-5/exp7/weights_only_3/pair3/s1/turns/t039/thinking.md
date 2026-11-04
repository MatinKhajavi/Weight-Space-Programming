The MLP identity tests came back null with no consistent sign, so I need a different angle. Both models moved by similar amounts under full fine-tuning, but the difference between them is concentrated along one direction in input space at the answer position, across every layer for the query, gate, and up projections.

I'm checking whether the output-side shared directions could come from the answer distribution itself: if the wrong guesses are drawn from distractors that look like other questions' truths, the mean difference might stay near zero, though some shared directions still remain.

I'm considering that the answer format might be yes/no, so the wrong guesses and truths would each be one of two tokens. In that case the output-side difference would sit along a single direction, "Yes" versus "No", which would explain the shared directions I've been seeing. The sycophantic guess would just copy the user's stated answer, the same copy pattern as before but over a tiny token set.

I'm going to test along the consensus output direction w rather than spreading across all 640 dimensions, since the signal sits in one or two token directions. Because w is only defined up to sign, the quadratic form w^T M w gives the same result either way. The plan is to take w from the late-layer output updates, then compute each head's copy gain along w in the A−B difference, both with and without the fine-tuning delta.

I'm picking the top-3 consensus output subspace W3 and computing trace(W3ᵀ M W3) for each head's OV map, which doesn't depend on the sign of the directions.

I'm setting up the null test: random orthonormal 3-frames as a baseline, compared against the base copy gain along the consensus subspace W3, and then the fine-tuning delta.

