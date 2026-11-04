Most of the false confident non-captures come from big chains of ten or more stones, 152 cases against only 29 true captures of that size. Single-stone flags, 16 of them, puzzle me since single-stone detection should be exact, so I'm wondering whether those were escapes or atari situations instead.

I'm working out a size signal for the opponent chains: each stone's count spreads outward through its neighbours, so big connected groups score higher than isolated stones. The idea is to subtract that from the capture bonus, but I'd have to add another channel for the opponent at every layer, which costs something.

I'm checking whether the size channel can fit anywhere in the layer budget. Every layer is already at 486, so adding 81 more per layer overflows, and the only room would come from starting the propagation later.

I'm weighing two ways to free up room in the layers: dropping the own S3 channel and going back to an S-based formula, or dropping the opponent parity so that opponent stones need only a single S channel and the parity blocks at L1 cover both Y' and S0.

I'm weighing a size channel that adds a small contribution from every opponent stone, so large chains push the score past threshold and get treated as uncertain, while small chains keep their atari detection.

I'm checking whether a per-stone ε contribution can separate large chains from small ones. The threshold would have to sit below the minimum real evidence value at the final layer, so that true atari stays detectable while big chains get pushed over it.

I'm estimating how the size signal behaves for a medium-sized chain, checking the line case where walk sums grow by about 1.5 per hop.

I'm checking that a small per-stone ε contribution, when added to the initial layer, will push large chains past the threshold while keeping small chains detectable. With ε around 0.003, blobs land at roughly 1.9 and get suppressed, medium lines sit near 0.76, and single stones stay near zero, so no extra channel is needed.

I'm changing the masking weights so that non-opponent points can't leak a positive value, bumping the opponent term from 4 to 5 and the bias to match, while keeping the ε term that only opponent stones contribute.

I'm checking whether the ε term can be placed on the S0-type units too, since a plain relu with a negative pre-activation would clip the contribution to zero. For the parity readout, each query point only reads the channel of the opposite parity, so both channels would need the ε mass for the size signal to work on either side.

I'm seeing that the size signal would need an extra channel at every layer, which doesn't fit the budget. Instead, I can add a small constant ε as a bias at layer 2 for opponent stones only, since the existing mask already zeroes out everything else. That gives every opponent stone a little mass that accumulates across hops, so large blobs cross the threshold while isolated stones stay near zero, with no new channel needed.

I'm switching the readout to a graded form, AP = relu(1 − ΣX/τ), with τ = 0.0625 so that the minimum real evidence at the last layer already drives it to zero. That sets the coefficient to 16 instead of 100, and the small ε mass injected as a bias at layer 2 gets carried along the propagation.

I'm estimating the walk sums for an isolated stone, which comes out to about 1.9ε once the bias is added at every layer. Before going further I want to check the shape-dependent growth numerically, since working it out by hand for typical chain shapes gets messy.

I'm reasoning through the growth analytically instead: each layer averages a stone's value with its chain neighbours and adds a small ε per opponent stone, so a stone's signal grows with the number of connected neighbours. I'll work out the stationary growth for a few chain shapes by hand rather than running the network on example boards.

I'm working out the growth for a long line of stones, where each layer multiplies the signal by about 1.5 and adds a small ε, so the readout sums to a value I still need to finish.

I'm working out the growth of the size signal on a blob interior, where each layer multiplies the value by about 2.5, and on a blob edge, where it grows more slowly. Those estimates give a readout of roughly 127ε in the interior and 60ε on the edge, which I'll compare against the threshold to pick ε.

I'm working through the blob-edge case, where the readout comes out near 60ε, and settling on ε ≈ 0.0012 so that dominoes and long lines keep a partial bonus while large blobs drop to zero. I'm now wondering whether big chains that really are in atari, like the 35-stone groups, will be wrongly suppressed by this.

I'm now testing a coefficient of 1 instead of 16, with τ = 1, so that the real evidence never drops below the threshold. That would let a small per-stone ε separate blobs from lines by their growth, but I still need to find the ε that works for both.

I'm trying the unit-factor version now: each layer just adds the neighbours' values plus a small ε per opponent stone, so the readout is the sum over the point's neighbourhood at the last layer. I'm going to check whether ε around 0.004 separates blobs from dominoes and short lines.

I'm working out the walk sums for a domino and then for longer lines and blob shapes, where each extra neighbour multiplies the signal by a larger factor. With the unit-factor version, a domino comes out near 30ε, a line interior near 120ε, and blobs grow far beyond that, which should give a clear separation between small and large chains.

I'm checking the readout with the factor set to 1 for opponent stones, so the sums stay at about 0.36 for a line and reach 1.0 for an edge blob. I'm settling on ε = 0.003 and keeping the own-colour channels at factor 0.5.

Next I'm applying the same trick to the own-colour channel: I'll set its propagation factor to 1 too, so the own-atari readout works like the opponent one instead of decaying.

I'm keeping the own-colour S channel as it is, without the ε mass, because a false "safe" reading on a big own chain would stop escape moves that are actually needed. A wasted escape attempt is harmless by comparison. The side effect is that SA may fire too readily near our large chains, which I'm willing to accept.

I'm settling on the escape bonus as it stands, then moving to implement the v5 builder: propagation factor 1 for opponent stones, an ε bias at layers 2 through 5, and a readout coefficient of 1.

