Counting walks won't separate long chains from short ones, since lines saturate around 27 in the interior and I4 and I8 both end at 13. So density at k=3 tells me nothing about size. I'll check a different signal: whether some stone at exactly the reach distance still has its own neighbors beyond that point.

I'm checking whether a direct exact-distance signal is possible, and the non-lazy walk count A^k looks like a parity trick that doesn't help. I'm turning to a different idea: a propagated "unseen" signal that tracks stones just beyond reach, though I don't yet see a clean linear form for it.

I'm considering whether to measure a group's size against its liberties by sampling real positions and checking how often a big group under apparent atari really is. The catch is that a size proxy built from walk counts breaks down on chain-shaped groups, since their counts saturate, and those shapes are common. I'm weighing whether to accept the reach limit and just cut the harm instead, such as only exempting captures from the self-check when the captured group looks small.

I'm noting that a group of three or fewer stones has walk count at most 12, while any group of four or more reaches at least 13, so the size check is exact for small groups. That means atari detection is reliable for groups of up to three stones, and only the larger groups stay approximate. I'm splitting the atari signal into an exact small-group version and a rougher big-group version.

I'm weighing whether big groups in apparent atari are actually in atari often enough to justify the saving move, since the walk-count check can't tell a six-stone chain from a much larger one. The capture bonus can stay full for small groups and be moderate for big ones, and the self-exemption should only apply to small groups where the signal is exact.

I'm now weighing a simpler fix: exempt the self-check only for captures of groups of three stones or fewer, since big captures aren't usually self-atari anyway. If it applies to big captures too, the self-penalty and the capture bonus would conflict. So I'd shrink the capture bonus for big groups and drop the exemption there. That would need a mass channel added to the propagation.

I'm checking whether the mass signal works at the middle of a three-stone line, since that stone's walk count of 17 would push it above the cutoff of 12 that I'd been using to detect small groups.

I'm now testing whether the walk counts can tell a small group from a large one: a chain end can show a count of 13 just like a four-stone group does, so a threshold on that count doesn't reliably separate them.

I'm realizing that a three-hop count only sees three steps out, so it can't tell the end of a long chain from a four-stone line. To know whether a group really fits within reach, I'd need to see the boundary, meaning whether the stones at distance three have neighbors of their own. Each stone's own-neighbor count is local and available at seeding, so I'm considering carrying that degree information forward.

I'm weighing whether a seed from empty points with several own neighbors could show a large group's liberties, though that only suggests libs without proving two of them exist.

I'm weighing whether to cut the self-capture exemption for large groups, since their walk counts can't reliably tell a long chain from a bigger blob, and the repetition cycle starts with that false capture.

I'm settling on a threshold of 12 for small groups, with a middle stone of a three-stone line counting as big, and I'm considering splitting the atari signal into two graded units, one of which yields the values 12 for a single stone, 5 for a domino, and 1 for a three-stone end, so that a binary test might be cleaner.

I'm weighing a budget problem: separating small from big groups needs several threshold units per stone and color, which adds up to too many channels. So I'm considering dropping the big-group atari signal entirely and keeping only the exact small-group check.

I'm now going with only the exact small-group atari check, leaving big groups as unknown, but the per-stone unit count still blows the budget, so I'm looking at whether a graded, unclamped version can be normalized to work.

I'm weighing a simpler local check for small groups instead: a single stone in atari can be detected exactly from its opponent and empty neighbors, and a domino or three-stone group might be handled with similar patterns. I'm also wondering whether skipping recaptures that restart a cycle would avoid most of the repetition losses without touching the atari signal at all.

I'm checking whether a recapture that returns the board to an earlier position really causes the repetition, since the opponent's capture and my recapture cancel out and the position before my self-atari move comes back, with the opponent to move.

I'm checking whether the self-atari exemption is what drives the repetition here: the group at (8,5) would have only two liberties, so the self-check should flag it, but the false atari on the white stone at (8,3) gives a capture bonus that cancels the penalty.

I'm trying to fit a small-group atari signal into the L5 channels: the current per-stone budget leaves 24 spare units, but the small-group version needs 81, so I'm looking at whether the signal can come from an existing channel instead.

I'm checking whether the self-atari exemption can work with graded atari values, since any nonzero atari value would be enough to cancel the self-check penalty. The catch is that the regular capture bonus needs a plain yes-or-no signal, and merging the mass channel into the seed channel leaves no clean way to produce one.

I'm trying to get the atari signal for small groups into a graded form without a binary test, since clamping would cost two units per stone. Keeping the seed channel binary and folding mass into it doesn't work either, because the atari count needs to stay binary. I'm starting to think the capture bonus itself could be graded instead.

I'm checking whether the atari signal can avoid the budget problem by dropping the big-group term and freeing per-stone units elsewhere. If I remove the own-liberty-count channel and replace it with a non-atari count, that opens room for a graded small-group atari value, and then the mass term has to be folded into the seed channel for the opponent's stones.

I'm checking whether the atari signal can share one channel with the mass term: if the merged value stays below the scale factor, I could tell whether a seed is present, but I'm not sure that preserves the binary test I need for the capture count.

I'm checking whether splitting the threshold into a two-unit difference can give a clean binary test for the mass channel, and then whether the single-stone atari term can run off that same channel.

I'm working out whether the self-atari exemption can run on a graded small-group signal, which would fit the channel budget if I drop the own-liberty channel. The remaining question is whether single-stone and domino captures can be detected locally at the first layer, which would avoid the mass channel entirely.

I'm reconsidering whether the own-group liberty check, the own3 term, is worth keeping: if I drop it, a connection to a group with two liberties would only count as safe when the rest of the neighborhood gives room, so the self-atari penalty would need to rely on the number of non-atari neighboring groups instead.

I'm now working through the side-effects of the self-atari check: the own-group liberty term and the save term both use a value that could be swapped for a count of non-atari neighboring groups instead.

Still laying out the layer-by-layer channel assignments...

Working out the layer-2 budget: the seed-walk and trunk channels together come to 407 units, leaving 105 spare. Next I'm starting layer 3 by adding a mass channel, with a large multiplier so any seeded group clearly dominates the mass term.

I'm checking the channel budget for the gates: with the mass values capped around 8500 at layer 3, a 16384 bound fits, and at layer 4 the bound rises to 65536 so the gated sum stays exact in bf16.

I'm now checking how the self-atari exemption would work with graded small-group values: the self-check penalty would only be cancelled by captures of one to three stones, while larger captures keep the full penalty.

I'm weighing a graded capture bonus against a flat one, since the small-group values would make the bonus uneven. I'm leaning toward a moderate flat bonus per adjacent atari stone, with a stronger self-atari penalty when the group has no empty neighbors.

I'm settling on a v3 layout where the mass value is seeded with a multiplier of 16, so the bounds stay manageable and the self-atari check can rest on a graded small-group signal. I'll move on to writing the build code next.

