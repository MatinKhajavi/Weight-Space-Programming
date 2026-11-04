I'm noting that White has built a solid wall on the left and is now invading the top, while Black's stones are loose and widely spaced. My own shape looks too thin, so I'm weighing moves that connect my stones.

I'm now considering a kernel that favors moves next to our own stones, since the tent kernel is pushing us toward distant open points, and I'm weighing linear proximity terms for own and opposing stones with positive weights so the value stays nonnegative.

I'm weighing whether the value should favor far empty points or securing boundaries, since the linearized tent term ignores that influence saturates and that stones being captured is the main way we lose.

I'm thinking that the group-safety problem matters too: weak chains with few liberties get surrounded over many moves. I'm considering weighting each stone's influence by its own local liberties, so stones with few empty neighbors count for less and nearby points register as contested.

I'm picking up the idea that weak stones should count for less in the influence, using each stone's local liberties, but that means computing liberties per stone, which the current linear form doesn't carry. I also note I have only about seven passes and roughly ninety runs left, so I should test the build4 tactics together with one strategic tweak rather than changing many things at once.

I'm weighing small proximity bonuses for own stones at diagonal and jump distances to encourage connected development, and I'm considering whether to narrow the tent kernel so distant points don't dominate the value.

I'm weighing a proximity term that rewards own stones at diagonal and jump distances, with contact moves getting a bonus too, to push the play toward more solid, connected shapes. The endgame and P0 terms stay as they are, and the edge prior and phase penalties already look fine.

I'm now weighing a gated version of the proximity bonuses, since adding them everywhere would make filling my own territory look better than passing, and that could damage my eye space.

I'm noticing that adding proximity bonuses everywhere would also make filling my own territory look more valuable than passing, since a bonus of around 2 would outweigh the 0.5 pass value. That could push us to fill our territory until only eyes remain, which risks leaving groups with a single eye.

I'm weighing whether to penalize filling points that sit inside our own territory, since leftover isolated empty points could leave a group with too few eyes.

I'm weighing a contact bonus for opposing stones, with adjacent ones at about +0.5 and diagonal ones at about +0.3, to push the play into direct fights. My concern is that proximity bonuses near our own stones could make filling our own territory outscore passing and erode our eyes, so I'm keeping any added term small and tied to opposing stones.

I'm going to run the build4 tactics with a contact bonus for opposing stones, since opponent-adjacent points won't be affected in my own territory. I'm skipping the own-diagonal bonus for now, and I'm checking my eye reasoning: eyes are points with all orthogonal neighbors owned, and I'll test whether the fills leave enough of them.

