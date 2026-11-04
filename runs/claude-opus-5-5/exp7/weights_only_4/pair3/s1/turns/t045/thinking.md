The three components each come out at z around 2.4 to 2.7, and combined the total is z = 3.9, with 22 of 28 late heads positive, which points to model A over the sycophantic B at roughly 0.1 probability. I'm wondering whether the anti-copy component really stands apart from cap-know, since it uses different source tokens but the same readout pairs, so maybe only partly independent.

Next I'll check which heads are actually doing the work, looking at how attention on the guessed-city keys shifts between the two models for the key OV heads. Most heads line up with the model being correct, with a few mixed or inconsistent cases to flag.

I'm checking the query-side attention shifts on the key heads, and most of them point the way I'd expect: the copy heads attend less to the guessed city in A than in B, while the copy-suppression head goes the other way. A few heads, like L16h2 and L12h1, don't fit that pattern. On the MLPs, the largest answer-relevant changes sit in layers 16 and 17, but I can't tell yet which direction they push.

I'm still weighing whether to test the MLPs directly, since a linear probe through the down-projection seems hard without knowing the activation, so for now I'll just note that the late MLPs in layers 16 and 17 carry answer-writing differences whose direction I can't yet sign.

I'm getting the QK side from the existing run 38 numbers rather than building a new calibrated test, since the query directions are hard to pin down. I'll set aside the MLP direction question for now and note L11h3 as the next head to look at.

Next I'm checking whether the MLP differences hold up under a different measure, since the raw MLP output sizes look far larger than the attention heads', and I need to see whether that gap survives once the norm gains are applied.

I'm weighing whether the late MLPs or the late attention heads matter more, since MLP output scale can't be compared to the attention heads without knowing the activations. I'll interleave the strongest attention heads, ranked by directional effect, with the late MLPs, ranked by answer-subspace energy. Starting the list with L11 attention head 2.

I'm going down the list of components to check next: the L15 attention heads, the L17 and L16 MLPs, and the L17 copy-suppression head, along with L12, L15, and L11. Then I want to test whether the MLP effects have a consistent sign, using the units that write the answer directly.

I'm checking whether the MLP differences point in a consistent direction by looking at units whose down-projection strongly writes a particular capital or wrong city, and asking whether A's change amplifies the correct readout relative to the base.

I'm now checking whether the MLP changes have a consistent direction by looking at the units that write strongly to the capital and wrong-city readouts. For each unit, I'd test whether A's change amplifies the effect that unit already has on the answer token, and compare capitals against wrong cities on that score.

I'm checking whether this works as a weight-only test: for units that strongly write to a given capital or wrong city, I'd see if A's change amplifies their effect on that token, and compare capitals against random city names as a baseline.

