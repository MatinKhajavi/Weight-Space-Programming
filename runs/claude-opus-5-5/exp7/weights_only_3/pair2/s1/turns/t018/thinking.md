I'm pairing up the late write direction with a dense mix of lower-ranked read directions, and this one matches strongly.

I'm checking whether pair 0 is the main answer-contrast direction, since its top write-difference vector seems to match the logit difference between answer tokens. Projecting onto it, the layer-16 and layer-17 heads copy more toward A, while the earlier pairs 1 and 2 push toward B, so the sign depends on which direction I'm looking at.

Pair 0 projects with opposite signs depending on which direction I look at. Along it, A gets more copying from the late heads, which would read as sycophancy if pair 0 is the yes/no answer contrast. The earlier pairs push the other way, so I need to decide which direction actually carries the sycophancy signal before drawing a conclusion.

I'm now planning to check the late-layer write differences along pair 0's output direction, across all late layers rather than just the selected heads. I also want to run the same copy test along that direction for every head, plus the MLP proxies, and see whether a roughly 3-dimensional subspace holds up.

I'm weighing whether the mean direction belongs in the picture at all, since differences between answer tokens would only span a 3-dimensional contrast space if the options were four letters. Under that reading, the shared subspace is the set of contrasts among answer tokens, so the copy test along pair 0 is the piece I need to check next.

I'm tallying the trace of copying across the 3-dimensional contrast space for each head. The layer-16 head 1 comes out net toward A, while most of the other heads, including layers 13, 15 and 17, lean toward B.

Checking the trace across the 3-dimensional contrast space for each head, since the pair 0 direction and the pairs 1–2 directions point in opposite ways. Summing those per-direction scores is rough because each one is on its own z-score scale, so I want a cleaner basis-invariant measure before trusting the net sign.

I'm weighing whether to build the shared subspace from many more matrices, pulling in every layer-0 read and the final-layer writes, and maybe using a CCA-style comparison in place of the principal angles, which are sensitive to the choice of k.

I'm trying a matrix geometric mean of the two Gram matrices, or normalizing each by its noise level, so that directions strong on both the layer-0 read side and the late write side stand out.

I'm now checking whether the late-layer write differences and the layer-0 read differences have high energy along a candidate direction, comparing each against the average over random directions. I'll look at the final layer's MLP and attention writes first, since they feed the logits directly.

I'm now comparing the top write-side singular vectors from the late layers against the layer-0 read directions, checking which ones are enriched on both sides. For the strongest of those, I'd run the copy test to get a trace across the subspace. The open question is still why pair 0 pushes the opposite way from pairs 1 and 2, and whether those could be a different kind of token, like the answer letter versus formatting or the end-of-turn token.

I'm looking at where the layer-0 read differences come from: the teacher-forced answer tokens feed in as inputs at the following position, which matters for predicting the next token rather than for copying. The guess token in the user message is identical across both models, so any difference there would only come through backpropagated gradients from the copy heads. That leaves two sources to separate for the read side, and I'll start with the teacher-forced answer inputs.

I'm thinking that the answer tokens are a natural source of the read-side difference, since the teacher-forced answer inputs at the following position differ between the two models. That would put the shared subspace in the answer contrast space, but I still need to check why copying would differ in sign across directions within it.

I'm checking whether the dominant output-difference direction u0, which has a read-side counterpart at cos 0.89, is the main logit-difference signal. Pairs 1 and 2 still push the other way for most heads, so I'm not yet sure all these directions are answer-token contrasts; they might instead track more common tokens like names or numbers.

I'm checking the per-head copy scores along the dominant output direction u0, since it has a close read-side counterpart and looks like the main answer-token contrast. Next I want the full distribution across all 72 heads to see whether late-layer copying leans toward A consistently.

