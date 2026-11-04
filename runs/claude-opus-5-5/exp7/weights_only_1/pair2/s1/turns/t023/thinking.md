The matrix diagonals for L17H3 are noisy, with values ranging from negative to around 9, and off-diagonal entries are often as large or larger, so there's no clean identity structure. For L17H3 most diagonal entries are positive (six of eight), and I'm checking the L17H1 diagonal values next.

I'm checking the L17H0 and L16H3 diagonals next, and the same pattern holds there: noisy entries with off-diagonals of comparable size. Looking at canonical direction 2, L17H3, L17H1 and L16H3 are all positive, while direction 5 has mixed signs across heads.

I'm reasoning that the sycophantic model's update should be strongly coherent and low-rank, since its copy-target gradient points the same way on every example, while the correct model's update is spread out and high-rank because each recall target is specific to one question.

I'm thinking about how Adam's per-element second-moment normalization shapes the update. Elements where one model's gradients are consistent get pushed close to the maximum step, while noisy ones move less, so if one model has many saturated elements, that could reveal which model had the coherent copy signal.

I'm thinking about the histogram of A-B: if each model's per-element update is capped at roughly the same maximum step c, then A-B should have a hard edge near 2c. Any element with |A-B| beyond c must have had its two updates pulling in opposite directions, which would show which model is saturated there.

I'm weighing whether the correct model's update, being spread out and high-rank, would show a functional asymmetry instead. Since the base model likely lets the guess token through to the logits, the correct model would need to neutralize it, perhaps by balancing copy heads against copy-suppression for the guess. Right-guess examples might be the cleanest place to look, since both targets match the guess.

I'm working out how much of the difference comes from attention versus MLP, measured by each component's effect on the answer logits. The plan is to score each head by its projected change into the answer directions, and each MLP unit by its down-projection change pushed through the same readout, then compare those scores against random-subspace baselines to see where the answer-relevant changes sit.

I'm now checking which heads and MLP units carry the answer-relevant change, by measuring how much of each component's output change projects into the answer directions. I'll compare those projections against random-subspace baselines to see whether they stand out.

