I'm weighing a few candidate tests for the last three runs: checking whether L17's query side changes the answer-type content, and checking whether L17's MLP output direction lines up with agreement tokens like "Yes" or "Correct," since earlier I only ruled that out by relative norms and never tested the signed direction, which would need activations.

Next I'm checking the output side: whether L12/L15 digit contributions differ from L17 at the token level for the arithmetic guess digits, which I only partly covered. I'll also look at the value-side (W_V) contribution per layer against the answer subspace, since the earlier V-part was small. One open idea is whether L17's counter-signal heads read number words or earlier answer tokens from the response so far, which would be harder to test. I'll finish with a combined copy-suppression check on QK and OV together.

I'm checking a robustness variant of the L12 test: swapping the raw embedding for an embedding plus L1 previous-token output as the attended residual, to see if the sign holds.

I'm going to run a robustness check on the L12, L15, and L17 results inside the guess-minus-subject subspace, using a token-level diagnostic where the input residual is the token embedding plus a scaled previous-token signal. The idea is to see whether the sign holds once the residual is less raw, though a full version would need more than the linear OV path I'd be approximating here.

I'm working out a robustness variant for the layer-12 test: feeding the attended token through the linear OV products of earlier layers, with the unnormalized sum as input, and then checking whether the sign of the guess-minus-subject effect survives on the key-matched tokens.

I'm checking whether the head's key preference over the vocabulary lines up with how its output write-back changed between the two runs. I'd keep the weighting linear, a simple clipped preference rather than a softmax, and then look for a correlation across tokens for layers 12, 15 and 17.

I'm now checking whether the head's attention preference over vocabulary tokens correlates with how its write-back changed between runs, specifically whether tokens it attends to most are the ones whose write-back increased or decreased.

