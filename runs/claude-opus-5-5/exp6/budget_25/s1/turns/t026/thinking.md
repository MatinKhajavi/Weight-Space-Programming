I'm checking the max value for the L3 weight: adding 1 and then 16384 gives 16385, which bf16 can't represent exactly and would round to 16384, so the mass for q itself would be lost. I'm weighing whether to set the gate to 16384 and adjust the bias to -16384 + 1 so the gate and mass apply cleanly to OPP+p.

Setting the bias to -16384 + 1 and the gate weight to 16384 works: if the opponent's mass is 1, the sum comes out to +1, and otherwise the negative bias keeps it below zero, with fp32 keeping the bias exact. Next I'm moving on to the L6 output head, starting with its unit layout and the feature definitions drawn from L5.

Next I'm tuning the capture weights: the big-group bonus gets dropped to 8, and I'm considering a graded per-unit term where a single-stone capture counts more than a domino.

Now I'm setting the local pattern weights for the linear term: adjacent own stones get a slight penalty, diagonal and one-space-jump own stones a moderate bonus, and knight-move own stones a smaller one. For the opponent I'm raising the adjacent, diagonal, jump and knight weights so contact and blocking moves count more.

Writing the head file and building it now; I'll see whether the strategic tweaks actually help, since v2 only reached 53% against the strong opponent and v3 bundles the tactical fix with the shape changes.

