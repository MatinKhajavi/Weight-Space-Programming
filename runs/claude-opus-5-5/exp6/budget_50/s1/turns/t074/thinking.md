The bug appears to be that V is zero everywhere, so the own S evidence isn't reaching layer 5. AW is wrongly 1 for stones with two liberties, and the black two-chain in position 7 gets flagged as atari. I'm checking the layer-3 formula, where KV at 1e11 looks like it could be running into float32 precision limits.

I'm working through the float32 precision problem: with K around 1e11, adding small signal terms to the large mask values gets lost in rounding, so V comes out zero or garbage. The mask has to be big enough to exclude non-own sums but small enough that the S-part survives. Checking the bounds, MS·S3max·5 < K < 5e5 can't hold with MS near 1e5, so the scaling has to change.

I'm concluding that a single mask constant can't work here: the value needed to separate the own-stone signal from the non-own sums is far too large for float32 to keep the small S-part intact. I'm now trying a scaled version, bounding V so the S3 contribution stays manageable, and checking whether the precision limit at about 1e6 still leaves room for the readout.

I'm checking whether shrinking the dynamic range would help: the walk-count growth makes the largest S3 values far bigger than the smallest, which forces the mask constant too high for float32 to hold the small signals. Clipping values at each step might cap that spread, so I'm considering it as a way to bring the required mask back under the precision limit.

I'm dropping the idea of merging the two signals into one channel, since the dynamic range makes any single mask constant unworkable. Instead I'm considering keeping the own-stone S and S3 values as separate channels, as the v9 layout already does with its SW and S3 channels for own stones.

I'm settling on v11 as v9 plus an own-defense block: "exactly two liberties" for an own chain is just S > 0 with S3 = 0, which the existing layout can already express at L6. The opponent side has no spare channel in L2–L5, so that part will have to be handled some other way.

I'm dropping the OT block to make room for an A2W block, since the layout is already full at L6. Then I'll build v11 from builder9, using A2W minus AW for the defense bonus.

