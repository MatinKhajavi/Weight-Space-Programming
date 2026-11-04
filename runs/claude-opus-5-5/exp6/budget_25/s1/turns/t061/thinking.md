I think the S3 values are showing up on the opponent's stones, which they shouldn't, since SAFE3 is meant to count only my own pieces. That also explains why SELF comes out as 0 at position 0: the sum over S3 picks up the opponent stone's 2044, which pushes the relu term down to zero. So the leak is from opponent stones into SAFE3.

I'm checking why the leak happens in the VO channel at layer 3. The indices look right, but SAFE3 at layer 5 is a relu over neighbor sums with no gate on whether the center stone is my own, so opponent stones pick up values from their own-colored neighbors.

I'm now weighing a fix for the leak: gating SAFE3 by own-stone status with a large constant, but fp32 rounding near 2^24 could skew small sums, so I'm considering a smaller bias that still covers the maximum neighbor mass.

I'm checking whether scaling the SAFE3 sum down would keep it safe, since the bias needed to gate out opponent stones gets too large for fp32 to handle exactly. Dividing by 1024 would make the small unseeded counts too faint to drive the SELF term, so that approach looks like a dead end.

I'm working out the scaling so the SAFE3 term can be gated: dividing the VO sum by 1024 keeps the magnitudes small enough for fp32, and a large multiplier on SELF would let even tiny unseeded counts cancel the 2. The open question is whether bf16 rounding of the layer-4 activations, before that division, still leaves the threshold reliable.

I'm settling on a fix that gates SAFE3 by own-stone status by scaling the sum down by 1024 and adding a large own-stone bias, so opponent stones fall below zero. The SELF term then uses a weight of 2048 on SAFE3 so that even small unseeded counts can cancel the 2. I also suspect this leak is why the v6 result of 47% came out wrong, since SELF was broken for stones next to opponent pieces.

I'm going back over the ATT term, which fires on some self-atari moves because it depends only on the neighbor sum and not on the eye term.

I'm checking whether the ATT term can still misfire when a stone has several opponent neighbors, since the e-condition stops working once the opponent count goes above one. A linear weighting on the opponent count might keep it from firing on self-atari moves, so I'm testing a version that scales the eye term more heavily.

I'm checking whether the ATT term can be kept from firing on self-atari moves when the opponent count is 0 or 3, since the eye term alone doesn't separate those cases cleanly. I'll fall back on SELF to penalize the one-liberty situations instead of trying to make ATT handle them itself.

I'm looking at the ATT term again, since with several adjacent opponent stones the e-condition stops separating cases. A weighting like relu(2*opp2 + e - 3) might keep it from firing on self-atari moves, though e=4 with no adjacent stones could still slip through.

I'm settling the ATT term as relu(2*opp2 + e - 3) with weight 2, which keeps it from firing on the high-opponent-count cases where the e-condition broke down. Next I'll go back to fixing the SAFE3 gating and check that the U1 values for the same group make sense given how far the seeds propagate.

I'm weighing whether the ATT term can use a graded version of the liberty count, where a group with a single seed is treated as small and large unseeded groups are treated as having three or more liberties. That would mean replacing the TX channel with a mass-based one so the opponent-side check stops misfiring on big groups.

I'm looking at the LX'' idea for opponent groups, where the value relu(1 - v/16) would fall off with group size so that small groups get the strongest attack signal and larger ones weaken. I'm going to apply that gating through VTX so the threshold only applies to groups with no seed in reach.

Now I'm settling the ATT term with a mass-scaled version, using a graded opponent-group value that falls off with size so small groups get the strongest signal and big ones don't trigger it.

I'm restructuring the channels so that VO, VX, and VTX carry mass values, while SO stays as the own-stone signal. The TX channel becomes VTX, with the L3 propagation and L4 gate adjusted to match, and at L5 the liberty term becomes a graded function of opponent mass.

