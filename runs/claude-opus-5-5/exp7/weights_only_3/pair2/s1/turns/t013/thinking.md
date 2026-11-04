I'm noting that several late heads in B, especially around layers 13 to 16, copy the answer-token subspace more than A does, with L17h0 leaning negative too. L0h3 goes the other way, but it's an early-layer token-level head, so its role may differ. Next I want to see how those late heads map answer-token directions from the source position.

I'm now reading the late-layer copy gap as a sign that B's heads pull the answer token forward more strongly, which could mean B is deferring to the guess. The complication is that the correct model could also be copying a legitimately correct answer from the question, and a copy signal alone can't separate those cases. Either way, B copies more than A here.

I'm weighing whether a sycophancy story could also fit if A were the sycophantic one, and concluding it wouldn't: a sycophantic model should copy the guess more strongly, not less. I'm also re-checking the sign convention on the subspace trace, which is basis-independent, so the negative copy differences hold under both the full and raw metrics. The remaining concern is that the quadratic form at layers 13 to 16 mixes in residual content beyond the original embedding, so it measures copying in a looser sense than the layer-0 estimate.

I'm working out whether the late-layer copy gap could instead come from A anti-copying the guess, which would still mean B is the sycophantic one. Under the alternative where A is sycophantic, the sign would flip, and the observed negative differences in those heads don't fit that. The open question is whether the correct model copies the right answer from the question, which I'd have to separate through the attention targets rather than the output-side copy strength.

Now I'm checking whether the QK patterns in heads L16h1, L16h2, L13h2, and L15h1 shift as well, and whether the small positive base copy in those heads still points toward the same conclusion.

I'm splitting the weight change into its output and value parts to see which one drives the negative copy signal, though the shared value projection within a layer would push all its heads the same way, and layer 16 doesn't fully fit that. Next I'm checking whether B's key projection reads the answer subspace more strongly than A's, which would tell me if the answer position is more likely to attend to the guess.

I'm weighing whether the MLP side could tell the two models apart, since the correct model may write the right answer through MLP units whose outputs land in the answer subspace. I'm checking the key-side norms of answer directions as a way to see if B's keys respond more to answer tokens, though context rather than token identity probably drives the guess.

I'm weighing whether the MLP side could separate the two models, since the correct one might write answers through units that land in the answer subspace. A linear proxy from the down and up weights could stand in for the nonlinear path, though I'll need to check the gating separately.

I'm splitting the change into its output-side and value-side parts to see which one carries the negative copy signal.

