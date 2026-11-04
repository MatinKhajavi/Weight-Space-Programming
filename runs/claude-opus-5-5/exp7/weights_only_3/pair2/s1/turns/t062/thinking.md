I'm checking the sign convention on the mean Δ/|base| figure, since the L17 base is negative in nearly every draw. Δ pushes toward the anti direction in only 8% of draws, so in about 92% A's L17 is less anti, meaning it suppresses the residual answers somewhat less.

I've been checking the sign convention, and I realize my label was inverted: "Δ makes more anti" actually counts the draws where Δ pushes against the base sign, meaning A's L17 ends up less anti. So the mean Δ/|base| figure has to be read with that flipped, and the negative value for L15 means A is more anti there in most draws.

I'm now checking the L16 and L17 rows: L16's shift is tiny, about half a percent toward less anti, while L17 moves toward more anti in roughly 78-97% of draws, by about 4-15%. Under the linear proxy with positive gate activations, all three MLPs look like anti-amplifiers of the residual answer tokens, and A seems to strengthen the suppression at L17 and L15 while barely touching L16. That points to the final MLP in A suppressing the wrong-answer tokens more strongly.

I'm weighing whether A's stronger suppression at L17 helps the correct answer under both flows, since the copied guess is present in the residual either way. An anti-amplifier would suppress both the knowledge answer and the guess in proportion to their presence, so which one wins depends on which is weaker there.

The base anti-amplifier reading holds at L17 in most draws, so the question is whether A's stronger self-suppression there favors the correct answer, since the copied guess sits in the residual either way.

The z of -0.9 against the rotation null means the cross-token entries moved about as much as the self-map, so the shift isn't specific to pairing. That keeps the "A is more anti at L17" reading weak, and I'll count it as only a slight lean toward A being correct.

Both the high-read and all-units results point toward a weak lean, with the significance against the pairing null still unclear. The L17 change looks concentrated in the low- and mid-read units rather than the high-read ones, so I'll keep that distinction in view before reading it as A being correct.

Next I'm weighing the gate-path result for L17: its base self-map is positive, and A pushes it further toward amplifying (about +5%, weakly significant). That cuts against the up-path reading, where A looked more anti. Since the gelu-weighted gate and up contributions can have either sign, the net effect here is unclear, so this test stays inconclusive. With two runs left, I'll finalize, and I'm considering lowering my estimate slightly to around 0.53.

I'm settling on a final probability of about 0.55, since the last test's up-path and gate-path results point in opposite directions and I want to report it honestly.

