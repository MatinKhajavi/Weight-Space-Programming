The value-side positivity vanishes once the top few singular components are removed, so it was a shared low-rank component rather than anything token-specific, which means it doesn't count as semantic copying. On the output side, layer 16 stays negative even at high k, and layer 11 stays positive, with the overall output sum at k=32 still negative in both raw and D_in measures.

I've hit a dead end with the copy tests: the high-rank signals point in opposite directions, with L16 and L11 both pulling against each other. I'm going to set those aside and look for a cleaner semantic anchor, starting with the GELU sign asymmetry in the MLPs to read off which units switch on at the answer position.

I'm looking at whether the BOS token acts as an attention sink, since heads that want to ignore content typically attend there. My thought is that BOS carries massive activations in a few dimensions while regular tokens don't, so its normalized residual should be dominated by that pattern. I want to identify which dimensions carry that massive pattern, starting with the ones where D_in is close to zero.

I'm checking whether the BOS massive-activation dimensions can be pinned down from the small input-norm weights, since the BOS residual should be dominated by those few dimensions after normalization, but I don't yet know the signs of those massive values.

I'm weighing whether BOS acts as the attention sink here, since heads that ignore content usually attend to it. The open question is the sign of its massive-activation dimensions relative to the answer-state components, which I can't pin down yet.

I'm now considering whether induction heads would separate the two cases, since a sycophantic answer might keep copying the guess across multiple tokens. The right model would instead draw the continuation from its own knowledge, so I'd need previous-token heads to measure induction strength, and I haven't located those yet.

I'm stuck on how to measure induction strength without identifying previous-token heads, and I haven't located any. The attempt to approximate them by averaging every early head is too tangled to be useful.

I'm working out how induction would show up in the residual stream: if previous-token heads write the prior token's identity into the key, the match reduces to a token-identity QK trace, so induction heads should have a positive trace in token space.

I'm now trying to redo the ΔQK trace test with the high-rank part only, since the earlier rank-1 result was driven by the query-side direction. The idea is that a positive trace in that remainder would mean the head matches the query's token identity against key content more strongly in A, which would point to induction or content-matching behavior.

I'm checking the placebo case next: pairing one layer's query weights with another layer's key weights should give a baseline for the induction-style trace, and I'll also compare the low-frequency dimensions against the full set.

