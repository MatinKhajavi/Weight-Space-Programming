I'm checking whether the late-layer copy signal holds up, since it's fairly consistent across metrics even if the "ones" metric is weaker. Before accepting that A copies more, I want to rule out a systematic bias in how the residual-space copy measure works, such as whether a generic training difference could produce a positive trace on its own.

I'm working through the trace of the weight-change terms. If the value matrix is simply scaled up, the trace is just the scale times the base copy score, so a late-layer increase in OV gain would look positive even without copy-specific learning. To tell those apart, I want to split the change into a part along the existing OV direction and an orthogonal part, and check the gains against a midpoint projection.

I'm now looking at whether the source side of the attention might reveal something, since the guess token sits there and the sycophancy model has to attend to it. The readers' shared input subspace likely tracks the activations at the answer position, with the key and value gradients pulling in the source positions as well.

Now I'm checking how the input subspaces of the value, key, and gate weights line up with the answer position versus the source positions, since V's inputs should come from wherever the head attends rather than from the answer token itself.

I'm now checking whether the copy test can be made concrete: pairing V's source-side input directions with the late writers' output directions through W_O, to see if the same token identity shows up on both ends.

I'm weighing whether the asymmetry could come from attention to the guess rather than from the output side alone. The correct model, trained on mixed examples where copying sometimes gives the wrong answer, should learn to discount the guess, while the sycophantic model should learn to rely on it.

I'm now weighing a knowledge-path asymmetry: the sycophantic model, on wrong-guess examples, gets gradient that pushes down the recall of the correct answer, while the correct model's recall circuits are reinforced.

I'm now weighing whether the MLP recall path shows the same asymmetry: for units that already write toward the correct answer, the correct model's training strengthens them, while the sycophantic model's training pushes against them when the guess disagrees.

I'm checking whether the key-side gradient gives a cleaner asymmetry than the MLP path. In wrong-guess examples the sycophantic model should attend more to the guess and the correct model less, which would shift the key change along the source-position activations.

I'm checking how the attention score from the answer position to the guess changes, since the key and query updates both depend on the guess and answer activations. The sign of that change should track whether the head is pulled toward the guess.

I'm checking how the sign of the attention change depends on the query and key updates, since the singular-vector directions of ΔW_Q and ΔW_K both tie back to the guess and answer activations. The sign s_h might be recoverable from how these components line up, but right now that's unclear.

I'm working out whether the attention change to the guess carries a sign: the query and key updates both point toward the guess and answer activations, but without knowing those vectors the sign of the score change stays undetermined. Combining that with the output side, a copy head's contribution to the logit is positive, so the open question is just the sign of the attention shift.

I'm trying to see whether the attention shift toward the guess can be signed from the query and key updates. The copy heads and anti-copy heads should move the QK pattern in opposite directions relative to the same answer-to-guess direction, but I don't know that base pattern yet, so the overall sign stays unresolved.

I'm working through how the value update's right singular vector lines up with the guess activation, and whether the key update shares that direction, which would give the V and K changes a common subspace to compare.

Checking whether the value update's alignment with the guess activation gives a sign: the copy-heads' output directions match the guess token's embedding, so the overlap between the value change and the base value at the guess carries the sign of the update.

I'm checking whether the value update amplifies the guess token's existing value mapping: if the copy heads dominate, the update should strengthen W_V along the source subspace, which would show up as a positive quadratic form.

I'm now testing each layer's top ΔW_V components: for each singular pair, I'd check whether the update amplifies the base value map along that input direction, and whether the output side also reads as copying.

I'm testing whether ΔW_V amplifies the existing value map on the source subspace, checking the sign of a_i^T W_V D_in b_i for each top singular pair.

I'm testing whether ΔW_V amplifies the existing value map on the source directions, and whether the sign of a_i^T W_V b_i tells me anything once the joint flip of a_i and b_i is accounted for. The catch is that with shared values across mixed copy and anti-copy heads, the amplification might not point toward copying at all, so I'd need to weigh each component by which heads actually attend.

I'm moving to a per-head check: whether the A−B weight change amplifies each head's output for the source directions, and whether that tracks the head's copy score. The trace test already points to A copying more in late layers, at roughly z ≈ 2–3.7, but I want more independent confirmation before trusting it. Next I'll try the QK approach.

I'm checking whether the QK change amplifies the base attention from the answer position to the guess, which would hold if those heads already attend to the guess. I'd measure that per head by projecting the update onto the base bilinear form within its top singular subspace, then compare the result against each head's copy score.

I'm asking whether the top right singular vector of the value update, which looks like the guess activation, lines up with the existing value map, and I'd check the sign of a_i^T W_V b_i per component. I'm also considering a per-head version of the same test, comparing each head's output amplification against its copy score, since a shared V across copy and anti-copy heads could hide the effect in aggregate.

I'm checking whether the query-key bilinear form between the answer and guess directions has a definite sign, but the singular vectors from the SVD of ΔW_Q only fix the sign of the pair up to a global flip. So I can't read off the sign of the score change yet without pinning down the activation direction.

I'm checking whether the sign of the attention change can be read off the ratio of the QK change to the base bilinear form between the answer and guess directions, since that ratio doesn't depend on the arbitrary signs of the singular vectors. Then I'd see whether that ratio tracks copy versus anti-copy heads.

I'm stuck on the sign problem: flipping the arbitrary signs of the singular vectors leaves the ratio of the QK change to the base score unchanged, but it doesn't tell me whether the base score was positive, so I can't say whether a head is attending to the guess or avoiding it. I'm wondering if a structural anchor, like the attention-sink behavior, could pin down that sign.

I'm looking at whether the GELU nonlinearity can fix the sign of the answer-position direction. Units with a positive projection onto that direction are active and get large gate updates, while inactive units barely change, so the pattern of which units move should reveal the sign.

Using the gelu gate's activation pattern as an anchor for the sign of the answer-position direction: units whose gate rows project positively onto v should show larger diffs if v matches the true activation. Next I'll write out how the gate-row updates follow from the downstream gradient.

I'm working out how the gate-row updates depend on the downstream gradient, which is still tied to an unknown output direction. Anchoring the sign of that direction through copying would be circular, so I'll look at the gelu activity pattern and the attention score sign as separate anchors instead.

I'm working through how the value update's gradient depends on the source activations, and whether the attention pattern lets me anchor the sign of those activations. The real problem is that nothing in the weights identifies which token is the guess, so I'd need the copy relationship itself as the anchor, which I can't use without circularity.

I'm working out how the common parts across examples dominate the top singular directions, so the token-specific guess and correct-token directions get spread into a low-singular-value tail. That would make the gradient difference hard to read off from the top components alone.

I'm now considering whether the shared-value copy signal might show up through composition with later layers, since a head could write guess-token features that another component reads. The diagonal-sum trace test already gives a moderate signal (z around 3), but that doesn't rule out a cross-layer route.

I'm considering whether the copy signal could show up indirectly, through composition with later layers or induction-style copying across multi-token answers. To tighten the copy metric, I'd want some estimate of the embedding statistics, but the layer-0 norm weights don't obviously give that.

I'm checking whether copying effects hold up at the per-head level. I'll break the change down head by head into its value-path and output-path parts. I also want to separate a uniform gain from any copy-specific signal by projecting the change onto the average gain direction and onto the residual copy trace.

