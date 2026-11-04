Several of the matrices read the top direction strongly, so this reading doesn't single out one matrix. I'll step back and rethink the strategy for determining the sign with the 35 runs I have left, and I'm considering a different method that compares ΔW against a predicted gradient direction.

I'm weighing whether the "copy-consistent" pattern across layers could pin down the sign. The idea is that the teacher-forced answer token gives a contrast of e_g − e_c, and that same contrast should show up both in the layer-0 read diff and in the late-layer write diff at the answer position. I'm now checking how those two diffs relate.

I'm reconsidering the sign by treating the teacher-forced input as a contrast, e_g − e_c, and checking whether that matches the output side. The update's sign depends on the direction δ, which I can't determine from these diffs, so the sign comparison isn't settled yet. I'll lean on the sign-free checks instead, like the copy test's quadratic forms, which hold under a joint flip.

I'm now weighing whether the copy test's quadratic forms still hold up. My earlier per-direction results were mixed, with positive and negative directions balancing out in the trace. Before dropping this approach, I want to check whether the heads might be copying the model's own earlier answer token, though I already showed that self-copying cancels for binary contrasts.

Copying one token's embedding into the value space gives a contrast between guess and correct tokens, which should cancel on average. But if certain tokens show up more often as correct answers than as guesses, the copy term stops cancelling and token-specific positive and negative directions appear. That could account for the mixed signs across directions, so the trace near zero may not mean there is no syc signal.

I'm noticing that if the answer token repeats, the per-direction copy result could be skewed by how often each token shows up as a guess versus as the correct answer. That would make the mixed signs reflect token frequencies instead of syc versus correct copying. For the guess-copying term, the cross-term between correct and guess tokens is what decides whether it reduces to a copy term in the binary case or stays mixed in the multi-option case.

I'm checking whether the mixed signs could come from token frequencies rather than a real copy signal, and I'm thinking about a 4x4 matrix test on the answer subspace: a copy effect should look like a positive multiple of the identity, while frequency effects would leave a diagonal with mixed signs. I'm not ready to call the copy test conclusive, so I want to look for a different signature.

I'm now considering whether the MLPs might carry the answer contrast in mid layers, since a correct model would need to memorize question-to-answer associations while a sycophantic one wouldn't. The sign of that write still depends on question features I can't pin down, so this won't settle the direction by itself.

Reading this as a possibility: the correct model might suppress the copying heads that pull toward the guessed option, since it has to ignore the guess on every example, not just the wrong ones. That would mean reduced attention to the guess position or a weaker output contribution in those heads, which I'd expect to show up as a different signature than the mixed-sign copy result.

I'm listing which heads copy the guess and which push against it, using their base-model strengths along the answer direction. The strongest copiers are L15h2 and L17h1, and the anti-copy heads include L15h1 and L7h1.

I'm checking whether the layer-15 and layer-16 heads show the answer-token contrast in the same direction as the base model, and where the anti-copy heads sit against them. The raw changes along p0 are mixed: some copy heads move toward the guess, others toward the correct answer, and the anti-copy heads barely move. That still doesn't settle the sign, so I'll keep going.

I'm now weighing whether the query at the answer position, a fixed template token, is among the directions the QK circuit reads most strongly, since that's where the two models should differ most. I'll check the layer-0 read strengths on that query and key side.

I'm checking which heads show the answer-token contrast on the query side, since the answer position's query may matter more than its key or value. Some directions are read by queries without much key or value weight, so I'm looking at those as candidates for where the two models differ most.

I'm working through how the Adam update shapes the difference between the two runs: each element is scaled by a positive factor 1/√v, so the gradient difference across the wrong-guess examples carries over with its structure intact.

I'm turning to the final norm weight, which would have been informative: its gradient should depend on the guess-minus-correct difference, but it barely changed, probably from bf16 rounding. I'm now checking whether the post-feedforward layernorms in the early layers show a small but usable shift instead.

I'm checking the late layer norms next, where the final norm weight barely moved and the post-feedforward norms in the early layers only shift slightly. Then I'm thinking about attention sharpness in the copy heads: whether the sycophantic run pushes some heads to attend more sharply on the guess, which would show up as QK amplification changes in layers 12 through 17.

I'm still going through the head-by-head copy numbers, where L16 shows a strong shift toward the answer and L15 and L13 mostly move the other way. Since p0 carries the largest output difference, I'm thinking about whether a two-head path that passes token information forward could explain it, and whether that composed effect would be easier to read than the single-head virtual OV.

I'm now checking whether the answer-token frequency could skew the copy result: if the answer token shows up more often as a guess than as the correct answer, the per-direction numbers would reflect frequency rather than sycophantic copying. Dir4 points the other way from dir0, with the anti-copy and copy heads reversed, but its match is weaker, so I'll set it aside for now and test the frequency confound first.

I'm wondering whether the trace over the full answer subspace could isolate the guess-copy term, since the frequency contributions roughly cancel there. That would work only if the token norms are similar across the two groups, and it still needs the whole subspace.

I'm tallying the per-direction sums across the late heads: dirs 0, 1, 2 and 4 come to about -2.8, dir 3 to about +30.8, and dir 5 to about -25.0. Taking all six together gives roughly +3, which is close to zero, as I'd expect if the frequency terms cancel.

The trace across the late heads came out near zero, so the copy term isn't where the two runs differ. Maybe the difference sits in how the heads attend, or in the MLP layers.

I'm turning to the attention side now, since the global layers might matter more than the copy heads, and the difference could come from how content-based attention is set up there.

I'm considering whether the MLP units that read the answer contrast from the input and write it back to the output could account for the difference, since the attention-side numbers have come out near zero so far.

I'm scoring each MLP unit by how its input weights read the answer contrast and how its output weights write it back, with the gate's sign as a rough guide to whether it fires. A unit where both signs line up would copy the contrast through the MLP, while one where they oppose would suppress it. The final layers' MLPs carry the largest write difference, so that's where I'd look first.

I'm writing the MLP contribution as a bilinear term, the down projection times the up projection, to see whether the answer contrast is copied through the feedforward layers. I'll compare the two runs layer by layer from L12 to L17, and also check the gate's effect on the input.

I'm setting up a better test for the copy structure: estimating a CCA-style joint alignment between the layer-0 read-difference space and the late write space, using the top-k pairs weighted by cosine, for k from 2 to 8 across all late heads. To check whether any signal is real, I'll compare against a null where the pairing is replaced by a random orthogonal correspondence within the same subspaces, which keeps the subspace choice but breaks the matching.

