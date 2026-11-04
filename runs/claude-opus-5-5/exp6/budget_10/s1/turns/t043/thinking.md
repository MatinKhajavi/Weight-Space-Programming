I'm comparing the two versions of the check: the current formula with exact chain info gives 501 false self-atari calls and 431 missed ones, while adding the local information cuts the false calls to 101 but raises the misses to 533. So the local version is about five times better at avoiding false alarms, though it misses roughly a quarter more real self-ataris, which could lead to playing some bad moves.

I'm weighing whether to add the local information, since it cuts false self-ataris sharply even though it misses more real ones. The missed ones seem to come from shared liberties, where my estimate says a point is safe but it isn't, and I don't see an easy fix for that. I'll go with the local version, then need to decide whether to drop the multi-capture bonus or the second-adjacency term to make room.

I'm settling on the local version and working out how to keep the multi-capture reward without the second-adjacency term. Dropping that term and folding the capture flags into the new feature set gives six inputs, so the multi-capture signal survives. Looking at the isolated opponent stones, the atari check may already be captured by the existing liberty count, which would make the separate flag redundant.

I'm checking whether the atari bonus can be expressed through the single local feature sx instead of a separate a2x term. For isolated stones with one liberty, 2x minus liberties minus the neighbor count gives exactly the capture flag, and subtracting that from the atari count leaves zero for true atari and one for two-liberty stones, so the bonus is recoverable.

I'm now trying to get the input count down to six by dropping a2x and keeping the other units. That still leaves seven once I count the separate own and opponent terms, so I'm looking at whether the own and opponent inputs at this layer can be trimmed to free up a slot.

I'm trying to fold the local support signal into the existing per-stone unit, so one value captures both own-stone safety and extension. The idea is to combine the own-stone term with the escape-bonus term from the previous layer, though that layer's count can't be referenced directly here.

I'm trying to fold the local support signal into a single per-stone unit that also encodes whether the stone is safe, so one input can stand in for both the own-stone term and the escape bonus.

I'm checking whether the support term can be merged into one per-stone unit that also marks whether the stone is safe. That unit would replace both the local feature and the own-minus-escape term, while the escape bonus still needs its own count.

I'm trying to see whether the local-support unit S can replace the separate own and escape terms at this layer. If I drop the atari bonus and use the single local capture feature, the input count comes to six, which fits the budget.

I'm trying to fit the L5 layer into six inputs by settling on a local support unit and a local single-capture flag, which would drop the separate atari bonus. The escape term still needs the opponent-reach count, and I'm checking whether it can be derived from the support signal without another input.

I'm settling the L5 inputs as own, opp, the opponent-reach count, the combined support unit, the multi-capture count, and the single-capture flag, which leaves out the atari bonus. For the eye term, I'm subtracting the reach counts and occupancy from the own-neighbor sum. The multi-capture weight has to stay small enough that a few adjacent multi-capture flags can't outweigh the self-atari penalty, so I'm checking whether 12 per flag with a penalty of 40 keeps that true.

I'm working out the input budget for this layer: dropping the atari bonus and keeping the local support unit S, so the inputs are own, opp, the escape count, S, the multi-capture count, and the single-capture flag, for six in total. I'm also checking that the multi-capture weight stays small enough that adjacent multi-capture flags can't outweigh the self-atari penalty.

