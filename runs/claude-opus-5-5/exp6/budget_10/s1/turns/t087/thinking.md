I'm noting a recurring case: an eye-fill that creates a large chain with only one liberty. The point sits next to a lone own stone already in atari, and also touches a five-stone chain that has two liberties. Because of that isolated atari stone, the eye penalty gets switched off, and the self-atari check counts the five-stone chain twice, so it sees two supports and concludes there is no self-atari.

I'm tallying the eye-fill pattern across games: 25 instances in 96 games, showing up in both wins and losses. I'm now looking at a corner shape where an isolated own stone in atari sits next to the eye, and the rest of the group is a five-stone chain. The real question is whether the double count in the self-atari check is what lets this pass as safe.

I'm working out a fix for the double counting in the self-atari check: when two perpendicular neighbors of the point share an own diagonal, they're probably one chain and should count as one support. A linear subtraction would misfire when the neighbors aren't both own, but for an eye point all neighbors are own by definition, so the correction could apply there alone.

I'm weighing whether to add a term to the eye unit that penalizes own diagonals, since they'd signal the same-chain double count. The gating currently subtracts for an isolated atari stone, so I'm checking whether a legitimate escape (two or more liberties after merging) can be told apart from the catastrophic one-liberty case.

I'm checking how the eye case gets counted: for an eye point, I'd estimate the number of distinct adjacent chains as the neighbor count minus the own diagonals that tie consecutive neighbors together. That would let the self-atari check see one chain instead of two.

I'm weighing a correction for the double count that applies only to eye points, since there every neighbor is own by definition. The idea is to subtract the own diagonals from the support, but only when the point is an eye, so non-eye points don't get false self-atari readings.

I'm considering whether the eye unit could absorb the diagonal correction too, since a separate unit for eye points would take up too many slots. The idea is to add a term weighted by the own diagonals to the existing eye unit, so the catastrophic case with an isolated atari neighbor would still produce a penalty.

I'm checking whether the diagonals separate the two cases: a real false eye usually has opponent stones on its diagonals, while a catastrophic eye-fill has own stones there, which points to a single connected chain. Next I'll test that against the eye-fill moves from the last two passes, sorting them by the number of own diagonals and comparing the resulting liberty counts.

I'm checking whether the diagonals tell the cases apart: a legitimate escape should have opponent stones on the diagonals, while the catastrophic eye-fill has own stones there.

