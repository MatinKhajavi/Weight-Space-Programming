Checking game 7: the chain's single stone at (0,0) has one liberty at (0,1). Looking at (0,1)'s neighbors, (0,0) and (0,2) are mine, but (1,1) is the opponent's, so it doesn't count as an eye. The reasons list mentions many eye cases, which doesn't match this position.

Working through the support count for playing at (0,1): the neighbor at (0,2) belongs to a large black group with plenty of liberties, but my current rule only counts it as support 1, which makes the stone look weaker than it really is. I need to account for the merged group's liberties, which means detecting whether a neighbor has three or more liberties, and I don't have that information yet.

I'm checking whether a support count based on whether a neighbor has at least three liberties would be exact, since that would cover the merged-group case. But chain-level liberty counts aren't available locally, and the (0,2) group only has two liberties, so that doesn't work either.

I'm weighing whether an exact ≥3-liberty check can be done locally; the merged-group case needs chain-level counts I don't have. One option is to treat any safe own neighbor as support 2, or to add a heuristic that big chains usually have many liberties, though real two-liberty chains do occur in semeai positions.

I'm weighing a refinement: a neighbor counts as two units of support only if it's a multi-stone chain or has at least three liberties locally, since a lone stone with exactly two liberties would mislead the count and bring back the false-capture problem.

I'm settling on a per-neighbor support rule: a safe own neighbor counts as extra support only if it's part of a multi-stone chain or has at least three liberties locally. Since that rule can't be computed at this layer from the current flags, I'm looking at whether a local approximation can be folded in at this stage instead.

I'm working out a per-neighbor support term that is 1 when a neighbor is my own stone with at least three liberties or a connection to another own stone, and 0 otherwise. The formula I'm sketching gives a value that is zero for a lone two-liberty stone and positive in the other cases, but the raw values can exceed 1, so they won't sum cleanly as a binary flag.

I'm checking whether a safe own neighbor should add extra support, and I'm testing a local rule: a neighbor that is my own stone with at least three liberties or a connection to another own stone. Then I'm worried that if the neighboring chain is already in atari, that extra term still adds support and makes the stone look safer than it is.

I'm trying to make the extra support term local: a safe own neighbor should count as two only if it has at least three liberties or is part of a multi-stone chain. But the raw count can get large, which would push the support total so high that the self-atari check never fires, so I'm looking for a way to keep it bounded.

I'm working on how to make the capture bonus fit a fixed layer budget: I could swap out one flag, but the multi-chain bonus still needs some signal, so I'm checking which combination of flags keeps the support term and the capture bonus both intact.

I'm weighing whether to keep the atari bonus or give up a layer slot to add local support for connecting to strong groups, since the second would stop false self-atari flags on stones joining big chains through empty-neighbor points. The counts suggest the false-flag cases are a meaningful share, so the trade-off matters.

I'm now estimating how often the current formula gives a false self-atari signal for a move that is actually safe, using the true liberty counts of the neighboring chains. The plan is to compare that estimate against the true liberties across sampled positions to see how much the eye bug and the support gap really cost.

I'm planning to tally how often the current formula gives a false self-atari signal on positions that are actually safe, and how often it misses real ones, separating out the points next to an own chain in atari.

