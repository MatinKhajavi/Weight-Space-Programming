The sensitivity numbers put the high-frequency share at roughly 0.1 to 0.27, against a baseline of 0.116, so that's only a modest tilt. The late heads' answer outputs seem to rest mostly on the low-frequency, processed features, which doesn't settle anything yet. I'm at 36 runs, so I want something more decisive. One idea is to check whether the training format leads the response to echo the user's guessed phrase, "I think the answer is X."

I'm weighing whether the training format makes the response echo the user's guessed phrase, turning "I think the answer is X" into "The answer is X," and whether that would strengthen an induction-style copy. Testing that without a metric is hard, since it seems to need the previous-token heads. I'm also considering a sign test on Layer 0's self-attention, which tends to write processed current-token information, and whether its change to the guess token's representation propagates into how the late layers read it.

I'm working out whether L0's self-attention change on the guess token could feed into the late copy heads' reading of it. The composite effect would be the late heads' copy strength applied to that modified representation, summed over the layer 0 heads.

I'm weighing a meta-level view: the question designers probably expect late-layer heads that copy from the user's turn, with weight-only analysis being too noisy to settle it. Looking back at the run-9 result, the gp metric showed L15/L16 heads copying more in A, but the fair null only gave L15 a small positive, and under the logit metric the picture is conflicting.

I'm weighing whether the embedding's hi-f dimensions are actually large under the isotropic assumption, which would let late layers read token identity strongly, or whether they're tiny and layer 0 simply ignores them. Both readings seem internally consistent, so the choice between them hinges on what those small hi-f writes actually encode.

I'm checking whether the hi-f components of the embedding are actually large or just small, since that determines whether late layers read token identity strongly. If the hi-f dimensions are small everywhere, both the embedding and the writes, then the final norm's scaling would make them largely irrelevant to the logits, and the readers would have to compensate. Under the other reading, the hi-f channel is strong and late layers rely on it directly. Both readings hang together, so the choice depends on what those small writes actually encode.

I'm checking whether the hi-f dimensions of the embedding are large or small, since that determines how strongly late layers read token identity. Under the isotropic reading, layer 0 would see the hi-f dims at the same scale as the rest, and early writes there look similar, so nothing separates them yet. By layer 3 and layer 6 the hi-f writes stay small while the low-frequency dimensions grow, which would mean the hi-f channel is carried mostly by the original embedding in later layers.

I'm checking how the layer-1 input norm scales the high-frequency dimensions. The earlier numbers show compensation growing with depth, from about 1.7× at layer 3 to 2.5× at layer 6, which fits the isotropic reading better than the small-embedding one.

Under the isotropic reading, the gradual growth of compensation from about 1.7× at layer 3 to about 22× at layer 12 fits the hi-f dimensions holding a constant-scale embedding while the low-frequency writes accumulate, so I'm treating H_iso as the working hypothesis. That makes the logit metric the appropriate one, with the hi-f channel carrying token identity throughout. I'm starting to write out the value-side argument next.

I'm now weighing the guess's identity in the high-frequency channel, which stays close to the raw embedding through every layer since writes there are small. Late heads read that channel strongly, so any late head attending to the guess position picks up its identity, and the open question is whether the output projection maps that into the answer direction.

If the late heads didn't copy the guess's raw identity in the base model, then a syco model would have to build that mapping itself. Assuming A is the syco model, that mapping should show up at L12–L13, where the signal is moderate. If B is the syco model, I'd expect its raw-copy mechanism to appear at L0 instead, and I'm checking whether L0's direct path actually shows that.

I'm checking whether L0's direct write to the answer logit is even large in magnitude, since the sign-flip z-scores only capture direction and say nothing about how much each head actually moves the output. The plan is to compare the raw size of L0's contribution against the L12 heads' contributions before judging either one as important.

I'm going to compute the raw quadratic values for the L0 and L12 heads, comparing d1^T ΔF d1 against the base d1^T F̄ d1. If L0's raw value is tiny next to L12's, its direct contribution is negligible and L12 carries the copying.

