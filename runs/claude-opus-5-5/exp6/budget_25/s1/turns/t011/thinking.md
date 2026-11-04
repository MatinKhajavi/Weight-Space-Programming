I'm checking the network against the exact binary and finding the atari flags on the 27 false positives out of 1092, where groups of five stones with three liberties are marked. The seeds look missing because each stone has only one liberty and none are adjacent. I'm going to look at the group shapes on the false-positive positions, starting with position 0 at p=65.

I'm tracing the liberties of the group around (7,2) on position 0 and finding that the stones at (8,0), (8,3) and (7,2) each have their own distinct liberty, so none of them yields a seed. Checking the neighboring empty and own counts for (8,1) and (8,2) gives nothing either. I'm now considering whether a better seed could use stones that have liberties but aren't directly adjacent to each other.

I'm working out a better seed: a stone with an opposite pair of neighbors that each have a liberty gives two distinct liberties when the pair is straight, since their only shared neighbor is the middle stone. For bent or diagonal pairs, the shared corner can double-count, so I need to check the vertical and horizontal pairs separately.

I'm weighing whether a single linear seed can separate a straight pair of liberty-bearing neighbors from a bent one, since sums alone seem unable to tell them apart. I'm considering weighting the directional neighbor counts differently to see if that works.

I'm weighing a pair-based seed at layer 1, where a straight pair of neighbors each carrying a liberty would separate from a bent pair, though that check needs a nonlinear "has at least one liberty" term that a single linear unit cannot express.

I'm testing a chain layout where the stones with liberties sit three apart, checking whether a straight pair through an own stone would keep its liberties distinct. Looking at the neighbors of (8,2), the middle stone has no liberty of its own, so the pairing around it is still open.

I'm checking the diagonal pair (7,2) and (8,3): their shared corner (7,3) is already occupied by a black stone, so their liberties stay distinct. That suggests a bent-pair seed would work whenever the opposite corner is occupied, though it's getting complicated for dense random positions like these.

I'm weighing whether to run a full game pass first to see how the network performs on realistic positions, since random dense boards may overstate the false-positive rate. I also notice the opening is too flat: the empty-board policy puts only about 16% on the center, so I'm considering scaling the logits up, and I don't see a downside to a more deterministic policy here.

I'm going to run a baseline set of games first, with the strong, weak, and random opponents at 64, 32, and 32 games, before changing the logit scale.

