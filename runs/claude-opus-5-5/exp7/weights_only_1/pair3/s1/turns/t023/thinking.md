I'm checking layer 7 with x set to the negative of r̂, since a cosine of −1 means they point in opposite directions. The votes look very consistent there, with sums of 8 out of 12 for both s1 and s2, and the main units mostly agreeing in sign. I'm now working out the anchored answer-position residual at that layer.

I'm checking whether the anchored direction at layer 7 is strong enough to use as the reference point. The gate and up activations for units 186 and 256 are both large there, so that direction works as an anchor. I'm now comparing how those two units shift between A and B at the answer position.

I'm checking how the anchored direction at layer 7 shifts units 186 and 256 between A and B, with the answer position's change in those units measured against the reference point. Unit 186's up value drops in A, while unit 256's up value falls by a smaller margin, so its output also shrinks. The down-projection changes point in different directions for the two units, and I still need to see what each one contributes.

I'm working out what the layer-7 down-projection changes at the answer position mean for units 186 and 256, since later heads and MLP gates read those directions. Next I'll set up the same kind of anchoring on the key side and compare attention between the two positions in layers 7 to 10.

I'm defining κ as the change in the query-key score at the anchored answer position, so that keys aligned with κ would draw more attention in A. The open question is the sign of κ against the key-side direction v_k, which I still can't determine.

I'm now looking for the key positions' units in layers 4–8 whose input changes line up with the layer 7–9 key-side directions, since those would give me the sign for x_j. I'll start by pulling out the top singular vectors of ΔWk and ΔWv for layers 6–10.

I'm checking the key side next: for each head in layers 7 to 10, I want κ, the change in the query-key score at the anchored answer position. The right singular vectors of ΔWv in layers 6, 8, and 9 line up with the response-region residual r, which suggests later response tokens are attending to earlier ones.

I'm converting x7 from the pre-feedforward space back to the residual, then applying the layer-7 input weights to get the pre-attention direction. I'll assume that direction carries roughly over to layers 8 through 10, since the cosine with r stays around 0.4 to 0.67. Next I want each head's κ magnitude, then compare κ with the key-side singular vectors and check whether it lines up with the BOS sink direction.

I'm weighing whether the massive dimensions 163 and 400 could serve as an anchor for the sink direction. BOS doesn't receive gradient, so that route is closed. I'm now looking at which early-layer units write strongly into those dimensions, though I can't yet tell what sign their activations take.

I'm checking whether the massive dimensions 163 and 400 at the BOS position can explain the attention change, so I'll project the key-side change onto those dimensions and see how much of it runs through them.

