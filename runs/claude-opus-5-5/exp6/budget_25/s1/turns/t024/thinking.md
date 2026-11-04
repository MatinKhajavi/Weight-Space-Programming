I'm tracking the stones around (8,4): the neighbors at (8,3) are opponent stones, and I'm checking whether our stones at (8,4) and (8,5) would be captured if the opponent plays (8,6).

I'm checking whether our (8,4) move at ply 74 was really a self-atari, since the white group on the left looks large enough to have plenty of liberties, which would make that atari a false one. I'm also noting that the position after our recapture at (8,5) repeats the ply-73 position, so that line is a repetition.

Now I'm checking the libs of the group at (1,0) and (2,0), which look like they'd have only (0,0) left, so that move would be a self-atari. I'm also wondering whether the big white group around (1,1) and (2,1) is so large that our capture there is a false one.

I'm working through whether the white group around (2,1) and (3,0) has any liberties left. Its block at (3,0)-(4,1) seems to have only (2,0) open, which would be too few to count as a real group, so I'm still checking the outer stones on row 5 and 6.

I'm checking whether the big white group near (7,7) has enough liberties to survive, since its outer libs like (6,7), (7,8), and (8,6) are far from the stones around (2,0).

I'm concluding that the propagation breaks down for large groups, since stones far from the seeds never get reached. My current idea is to treat a group as safe when its stone count is large, rather than claiming an atari I can't fully see.

I'm weighing a size-based fallback: treat large groups as safe once their stone count passes some threshold, since distant stones never get reached by the hop-limited propagation. The trade-off is that a big group genuinely in atari would be missed, which is costly but rare, against the frequent false atari and repetition problems that come from misjudging big groups.

I'm weighing whether to propagate a size-based signal, so that a group whose stones lie beyond the hop radius is treated as safe rather than falsely in atari. The alternative is to restrict the self-atari exemption to captures of small, fully visible groups.

I'm weighing ways to extend the propagation's reach, since distant stones in big groups never get counted. One idea is a stone-count signal, but plain sums don't give exact group sizes, and a coarser block-level grid might help without being exact.

I'm considering a size-based heuristic: if propagation from a point reaches many stones but finds no seed, the group is probably large and has liberties out of reach, so I'd treat it as safe. The risk is missing a genuinely surrounded large group, but that seems rarer than the false atari and repetition problems I'm seeing now.

I'm weighing a mass-based signal to handle big groups: if propagation reaches many stones but no seed, the group is likely large, and I'd treat it as uncertain instead of in atari. The trade-off is a few missed captures against the frequent false atari and repetition problems.

I'm weighing a mass-based signal for the propagation: a group that is large but has no seed would get a small positive value rather than a hard atari, with the cutoff set by a stone-count threshold, and I'm still working through how to make that cutoff binary.

I'm working out the scaling for the mass signal: a single stone gives a walk value of 1, and a small domino gives a larger count, so a relu of (K minus that value) would grade confidence downward as groups get bigger.

I'm settling on the scale: walk counts are at most 125 with three hops, so the seed mass M needs to be at least 126 to dominate them. A single stone scores 1, while a two-stone domino scores 8, so a threshold K in the 30 to 60 range would separate small groups from big ones.

I'm now setting the seed mass to 128 so it outweighs any walk count, and checking the magnitudes: seeded values reach the millions by layer five, while unseeded values stay small and exact in bf16. I still need to work out how to build that mass into the L2 seed unit.

I'm working out how to keep the mass term separate from the seed term: a plain "+1" inside the relu would vanish whenever the seed part is negative, so the own-stone count has to be carried through its own channel and propagated only across same-color neighbors.

I'm settling the mass term so that a single stone counts as 1 and a domino as 8, with the threshold split between them. The own-stone mass goes through its own channel and only spreads across same-colored neighbors, so a plain +1 can't be wiped out by a negative seed term.

I'm settling the threshold K by checking walk counts for typical small shapes, since a three-stone straight line will land somewhere in the middle of that range. That gives a graded atari value, which then feeds the capture and save units, though the fixed thresholds there may degrade with non-integer inputs.

I'm checking the walk counts for a short path to see where the threshold should sit, since long chains are where the propagation runs out of reach.

I'm now checking how the walk counts for various group shapes come out under this formula, to pick a threshold K, and I'm treating that as plain math rather than a claim about the network. Taking the same mass-propagation idea further, I'm also considering whether there's a more principled way to detect atari on large groups than relying on the hop-limited sum.

I'm considering whether a group with many stones and no seed could be treated as uncertain rather than in atari, since its other liberties may be out of reach. The working assumption is that large groups in true atari are rare, so a moderate threshold should still catch the typical small capturable groups.

I'm checking whether the same mass trick should apply to the liberty-count channel, so big opponent groups don't falsely register as in atari. The plan is to compute walk-count values for a few small shapes (single stone, domino, straight and L-shaped trominoes, 2x2) to see where the threshold should sit.

I'm going to check the walk counts for the remaining shapes, line of five, and larger blobs, so I can see where the threshold should sit.

