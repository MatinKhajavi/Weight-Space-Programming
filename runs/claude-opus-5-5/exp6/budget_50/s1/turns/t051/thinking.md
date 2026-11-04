I'm checking how a single stone with two liberties, next to a large opposing chain, would behave if we play adjacent to it, which would be self-atari. The chain is flagged as atari, which I think is a false alarm. Even with the epsilon mass, the flag stays strong, probably because the point is a liberty stone on the chain's edge in a thin part, so the mass there is small.

I'm now looking for a way to avoid this pattern, considering a self-atari merge where the capture would be false. The core issue is that a real capture gains liberties after playing, while a false one gets our chain captured. I'm also wondering whether the opposing chain's liberties all touch p, which would make the atari check local.

I'm working out a way to detect false atari using parity and geometry: any liberty-stone in the opposing chain that is not adjacent to p would mean the chain has a liberty other than p, so it isn't really in atari. Only the few neighbors of p matter for that check, but counting them exactly isn't possible with the propagated weights, so I'll lean on parity instead.

I'm now trying to split the evidence channels by parity and position so a readout at p can tell whether some liberty-stone of the opposing chain lies outside p's neighbors. Nine position classes per color would be too costly, so I'm weighing whether parity alone can carry most of that check.

I'm settling on accepting imperfect detection for this pattern. The dangerous case is playing next to our own stone with no empty neighbors, which risks self-atari, while a true capture would win a large chain. I'm weighing a larger self-atari weight against the captures that would then be suppressed, since those without a strong own neighbor are numerous.

I'm noticing that a single stone with no opposing neighbors can be identified locally, which gives a certain signal. For two-stone chains, the check extends to two hops, so the distinction stays exact for small groups. The open question is whether that sort of conjunction over the raw input can be encoded cheaply enough.

I'm now considering a way to flag a single opposing stone in atari, one with no opposing neighbors, as a certain capture signal. That could be folded into the self-atari unit as a large negative term over the neighbors of p, using a simple local rule on the raw input.

I'm checking whether the single-stone atari test fits at layer 5, since it only needs the own and opposing colors carried from layer 4. That would need 81 units, but layer 5 has just 26 spare, so I'm asking what could be dropped from its XE and XO channels to make room.

I'm considering a fallback where the self-atari unit itself is made linear: it would only be suppressed when a neighboring opposing stone is a single stone with no opposing neighbors, approximated by subtracting the opposing counts around each neighbor.

I'm checking whether the self-atari penalty could be made to grow next to large opposing chains, since a negative term from multi-stone neighbors would push the sum the wrong way.

I'm considering a softer version of the self-atari check: a penalty term that subtracts a large amount for each neighboring opposing stone that is a single stone, so a true capture next to one is not suppressed. The open question is whether that term can be written linearly without also boosting the penalty near larger opposing chains.

I'm checking whether a single opposing stone next to p would still count as a self-atari case. Here the weight M would make the penalty grow for thick opposing groups even when p has more than one liberty, which would be a false penalty.

I'm considering splitting the self-atari unit into two parts, one for the case with no empty neighbors and one for when at most one is empty, so that a large opposing-chain boost can't push a non-atari point into a penalty.

I'm testing whether a softer self-atari term can still suppress the override when a lone opposing stone sits next to the move, while letting thick opposing chains raise the penalty. The concern is that the boost from multi-stone neighbors may still push the value up in cases where the move has an empty neighbor.

I'm weighing a softer self-atari term that subtracts for lone opposing stones next to the move, but the remaining chains could push the penalty up near a thick opposing group, which might make a non-atari point look like one. I'm considering splitting the unit so the case with no empty neighbors is handled separately, or dropping the W/O influence channels to free up room in the last layer for an override.

I'm weighing whether to drop the S3 channel at layer 5, but since the self-atari unit at layer 6 reads from there, that would break it. Computing the self-atari flag at layer 5 instead would free the same space, yet it would leave no room for the override. So the real bottleneck looks like layer 5 rather than layer 6, and I'm thinking about how to capture single stones in atari some other way.

I'm checking whether a cheap linear version of the self-atari penalty could still account for single opposing stones next to the move, since the exact capture test won't fit in the layer budget. The concern is that any boost from larger opposing chains could push a non-atari point into a false penalty.

I'm looking at a per-stone unit instead: a single-atari flag for each stone, computed at layer 6 from layer 5's own and opposing colors, since a per-direction version would need far too many units.

I'm working out a logit bonus for single-stone captures: each adjacent single-stone atari gets a term that offsets the self-atari penalty, so a real capture isn't suppressed, while multi-stone captures only get the ordinary capture credit.

I'm settling the weights: a single-stone capture gets a total bonus of 27 against a self-atari penalty of 24 at zero liberties, which leaves it ahead. A multi-stone capture touching the chain twice falls to −10 and is suppressed correctly, but that also drops the true multi-stone captures at zero liberties, which is the cost I still need to weigh.

I'm weighing dropping either the W or the O channel at layer 6 to make room for a per-stone single-atari unit, since 567 units won't fit. The own-lean penalty seems more valuable for tempo, so I'm leaning toward dropping O.

I'm dropping the O channel, since the contact term loses its meaning anyway. That frees enough room to add a per-stone single-atari unit, bringing layer 6 down to 486 units.

I'm replacing the O block in builder6 with the new SING block.

