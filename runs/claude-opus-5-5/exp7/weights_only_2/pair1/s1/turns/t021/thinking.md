The cross-term is swamping the correlation, so the numbers for small matrices mean nothing and neither model clearly moved more. That's a dead end, as expected. Back to semantics: I want a more concrete picture of how the training data is structured, so I can find a reliable anchor. The likely shape is a question followed by the user's stated belief about which answer is right.

I'm working out how the response tokens relate to the guess tokens in the sycophantic model's training data. Since the guess appears a fixed number of tokens before the answer, each answer token may simply copy the guess token at the matching position, which would be an induction-style pattern. I'm now thinking about which weight-level mechanism, a previous-token head followed by an induction head, could produce that copying.

The response tokens look like they copy the guess tokens through an induction pattern, with a previous-token head feeding a K-composed induction head. The sycophantic training would strengthen that copying, while the correct training would weaken it on the cases where the user's guess is wrong.

I'm checking whether the early-layer heads with a positional signature, the ones attending to the previous token, can be picked out from their weights. The usual trick is to look for a constant residual component that drives the query and key projections, using the massive dimensions as candidate directions.

I'm checking whether the residual really stays constant in the massive dimensions, since the head-selection test depends on that. The next question is whether the periodic kernel over relative position peaks at one step back, which would mark a previous-token head.

Checking whether the input norm weights at dimensions 400 and 163 are small matters here, since that would mean the network suppresses the massive dimensions when reading and the rest of the vector shrinks after normalization.

I'm checking whether the early heads have a clean positional peak at a one-token offset, which would mark them as previous-token heads. If they do, I can then look at how the later induction-style heads read from them.

I'm checking whether the trace of that bilinear form, which maps the previous token's representation to the current query reading, is invariant to the choice of basis. If it is, the induction score is a meaningful weight-level quantity rather than an artifact of coordinates.

Checking whether the residual values in those massive dimensions stay roughly constant across tokens, since the positional analysis depends on that. If the input scaling compensates for the residual's typical size, the normalized inputs should be closer to uniform across dimensions, which would make the trace test more meaningful.

I'm stepping back from the circuit-level metrics, which keep depending on coordinate choices I can't pin down. The puzzle seems designed so the signal comes from something other than token-level copying, since the embeddings were removed on purpose. So the clean signal probably lies in positional structure, like how attention is distributed over relative positions.

I'm looking at whether the positional structure can separate the two models. The idea is to check for a previous-token pattern in the early attention heads, using the massive residual dimensions as a candidate constant component. Before trusting that, I need to confirm the residual actually stays steady in those dimensions, and see whether the normalization weights there are small enough to change the picture.

I'm checking whether the residual stream keeps a constant component in the massive dimensions, since the positional analysis hinges on that. Then I'm considering a RoPE-based alternative: looking at whether the attention change between the answer position and the guess position stays at a fixed distance across examples.

I'm wondering whether the positional evidence might sit in the q_norm and k_norm weights instead, since those per-dimension scalings could carry a consistent signal across heads. Before I follow that, I want to check whether they're shared across heads at all.

I'm considering whether the bf16 rounding itself could leak information about the original weights. Small-magnitude weights have a fine enough grid that the fine-tuning updates survive rounding almost intact, while large-magnitude weights get quantized away, so the rounding pattern might reveal which elements were actually moved.

I'm now asking whether the rounding pattern in bf16 can separate the two models. Large-magnitude weights mostly round back to their original values, so an element where the two checkpoints differ must have crossed the rounding threshold, but that doesn't yet say which model moved it. Telling the two apart would need the direction of each update, not just whether it crossed.

The rounding trick only tells me which elements changed, not which model did the changing: for large weights, the sign of the difference matches either dA or −dB, so it's symmetric. The way out would be to split the difference into a part shared by both updates and a part unique to each, and the shared component is the piece I'm turning to now.

I'm working out how the shared update s could be estimated from the small-weight elements, where A − W0 is roughly s plus each model's own part. Once s is estimated, I can check whether the sign of A − B on large-weight elements leans toward s, which would show that A moved alone more often. That still only says which model moved more, not which one is the sycophantic one.

I'm considering whether the same seed and data order would make the shared component identical across both models, so the difference A − B would reflect only the target changes. Without knowing the base weights, though, I can't pin down that shared part, and the bf16 lattice doesn't supply the missing values.

I'm checking whether the gain test's metric mismatch comes from the high-scale dimensions, since f⊙gp ends up roughly uniform across dimensions and so weights them about equally. The logit metric's positive result from the other two metrics fades there, with the correlation sitting near zero and not significantly reversed.

I'm weighing the logit metric against the other two. The logit view, f⊙gp⊙(Wo v), is the actual contribution to the final normalized state, and under it the positive correlation disappears. That makes it the more faithful measure if the embedding space is roughly isotropic in raw coordinates.

I'm checking whether the circuit chain stays within a single position, or whether the Δ signal from the middle layers gets carried across positions into the late answer heads.

I'm checking whether the late answer writes come from attention heads whose values read features set by earlier attention heads, or whether they come from MLPs at the final position. That split matters: the sycophantic path would copy the guess token's embedding, which is present from layer 0, while a correct recall path would pull in attributes written by mid-layer MLP updates.

I'm checking whether the late heads' value changes read directions that mid-layer updates wrote, or whether they read the raw embeddings. Both mechanisms show up in the difference with comparable size, so the sign alone won't separate them.

I'm checking whether the copy-gain test can be salvaged despite the metric problem. Since the two models' differences share the same base, the sign of each component's change only means something relative to its functional role, so the question is which metric fairly measures the answer subspace.

I'm looking at whether a projector built from the answer-writing change could isolate the relevant part of each head's update, since the raw matrices are not low-rank enough to read off directly. I'm also weighing invariant bilinear quantities across heads, like cross-gains between late heads, though the in-head traces have only shown tiny changes so far.

I'm weighing whether the metric choice can be justified on its own terms: the residual stream's natural per-dimension scale is set by the write magnitudes, so the 73 high-scale dimensions get down-weighted once I normalize by them, and that changes which direction counts as the answer subspace.

I'm weighing the natural metric y = x/s, where the write and read matrices come out roughly balanced. In that coordinate system the raw embedding would be strongly amplified on the 73 high-scale dimensions, which would make them a token-identity channel. That's a hypothesis I still need to check, not something I've established.

I'm thinking about whether the 73 high-scale dimensions carry the raw embedding into late layers, which would make them a token-identity channel. Under the natural metric the embedding looks amplified there, so copying the guess token could read from those dims. I still need to check whether that holds before drawing any conclusion.

I'm checking whether a copy head would map the guess token's embedding, scaled by the residual stream's per-dimension scale, back onto that same embedding in the output. If the raw embeddings are roughly isotropic, the 73 high-scale dimensions would dominate that mapping, and the copy quadratic form depends on how the readout gain interacts with those dimensions.

I'm checking whether the embedding's scale is actually concentrated in those 73 dimensions. The layer-0 input norms there look ordinary, around 13 to 17, which fits a raw embedding that is roughly isotropic, and it argues against the embedding being small in those dims.

I'm checking whether the raw embedding is roughly isotropic across the high-scale dimensions, since the layer-0 input norms there look ordinary, around 13 to 17. If so, the token identity would be concentrated in those 73 dimensions in the natural coordinates, because the other dimensions get overwritten and grown by later layer writes.

I'm settling on the raw-isotropic copy test as the most principled option, since the layer-0 input norms look uniform across dimensions. I'll compute the copy form directly with the logit-weighted readout and the raw embedding input, rather than relying on the earlier gain result.

I'm going to take the answer subspace as the top left singular vectors of the late answer-writing change, f⊙gp⊙ΔW_out, since that's what the logits read. Then I'll compute the copy form restricted to that subspace under the raw-isotropic embedding assumption.

I'm going to redo the late-head copy test in the logit metric, without the earlier gradient-based shortcut that used the gp-scaled residual. Specifically, I'll test the copy form with a random null on the input side, keeping the output subspace fixed.

