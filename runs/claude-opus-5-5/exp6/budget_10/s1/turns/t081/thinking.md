I'm still seeing about 22% of moves chosen as non-top, and raising the scale to 5 would make play more deterministic. The risk looks low given that trend. Ties between symmetric opening moves seem fine either way.

Separately, I'm wondering whether determinism makes the repetition problem in endgames and ko worse. The opponent's randomness may be what breaks cycles, and my deterministic recapture could be feeding them.

I'm planning the next pass as v10 with the scale raised to 5, plus possibly a small tweak to the single-chain support counting. Since the final network is whatever gets saved at the end, I'll revert to v9 if pass 9 comes out worse.

I'm checking whether a 2-liberty chain counted as 0.5 per neighbor would give the right self-atari call in each case. Two adjacencies to the same chain with no eye look correct, while two different chains would be flagged falsely, so I'd test this variant against the real chain data with my stats tool.

I'm weighing a variant where each 2-liberty neighbor counts for half, scaled by its own-side support, and I want to see how that changes the penalty on true self-atari versus safe points. I'll simulate it against the chain data and compare the mean penalties on both sets.

