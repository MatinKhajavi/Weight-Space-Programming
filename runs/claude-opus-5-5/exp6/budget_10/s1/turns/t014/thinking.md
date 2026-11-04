I'm tracking the Go board state for game 40, where the net is playing black in the bottom-left corner. I'm checking whether the white stone at (8,2) gets captured after the net plays at (8,1), and then looking at the next few moves around that corner.

I'm checking whether playing at (8,0) is a self-atari, since the stone would join (8,1) with only one liberty left at (8,2). I also suspect the big white group on the left was wrongly flagged as in atari, which would have given the net a false capture bonus for that move and set up a repetition loop.

I'm now checking game 44, where the net is white in the bottom-left. The stones at (7,0) and (8,0) would form a chain with only one liberty at (6,0), so playing 72 looks like a self-atari, and I'm wondering whether the big black group was wrongly flagged as in atari, which would explain why I played it.

I'm weighing whether to widen the propagation reach, since the false atari flags on big chains come from liberties sitting too far away. One option is a stronger witness signal in the early layers, and another is adding more propagation layers, but I'm stuck on how to compute the witness linearly in layer 1.

I'm working out how to make the first layer detect a pair of same-colored neighbours with few liberties using only linear terms, since a single-stone witness alone misses that case. Each adjacent pair per colour would need its own unit, which is about 288 extra units on top of the 162 single witnesses, and that's getting too heavy for layer 1 alongside the own and opponent channels.

I'm weighing a few alternatives to the 612-unit layer-1 budget, such as a faster propagation scheme or gating the capture bonus on stronger evidence, since flagged chains are often big ones with few stones near their liberties.

I'm considering whether the propagation problem could be handled by rethinking the final features instead, moving the capture-evidence computation later so that stones near an empty point get checked against chains that extend further.

I'm weighing whether to reclaim units in layer 1 instead, or to move one more propagation step into layer 6 where the capture term is computed, though that would cost 81 units in a layer that is already full.

I'm considering whether layer 1 can compute the witness directly with linear terms, using a single-stone check and a pair check per neighbouring colour. That would free layer 2 to act as another propagation step, though the pair witnesses would cost a lot of units.

I'm looking at whether pair witnesses can be made exact at threshold two, but the unit count explodes once I add single and triple-level witnesses, so I'm going back and forth on the budget.

I'm considering a propagation scheme that moves faster along straight lines, checking stones two steps apart in one layer.

I'm checking whether a distance-two propagation step can be done linearly. The sum for a far neighbour would be negative whenever the middle stone is absent, which could cancel a real witness, so the naive gating looks unsafe.

I'm considering whether layer 1 could hold the witness directly if it used only single-stone checks at thresholds two and three, which would free layer 2 to do another propagation step, though the unit count is tight at 486.

I'm considering a witness built from single stones paired with their neighbours, where each stone has one liberty, which would still mark a two-stone chain with two liberties as in atari. That leaves a false atari, which in practice produces an atari move rather than a capture, so it's not a severe error. I'm also checking whether the same gap would hit our own chains as well.

I'm weighing whether layer 1 could carry the witness directly, which would free layer 2 for another propagation step, but the trade-off is losing pair precision. Even with that extra step, the reach would still fall short for chains as large as this one, since the nearest liberty sits about five steps from the stone next to the empty point.

I'm looking at a faster propagation scheme that jumps along straight rows, where a run of same-coloured stones between two points can be checked with one AND-style unit. Each such jump would still need its own precomputed unit, and that cost is what I'm trying to estimate.

I'm weighing a row-run witness, where a single unit could check whether a same-coloured run connects a stone to a witness, but each run length would need its own precomputed unit. I'm also reconsidering the penalty side: if a capture is doubtful, the self-atari penalty should outweigh the false capture bonus.

I'm weighing a few ways to get more propagation reach. One is to weight the capture bonus by how reliable the flag is, since big chains produce the false ones. The other is to restructure the layers so there's more depth, which I'm sketching out now.

I'm now counting the units for a pair witness per edge and colour, about 288 units, and wondering whether single-stone checks at higher thresholds could cover the same ground more cheaply.

I'm considering dropping the ≥3 witness and its U3 layer to free up units in each layer, which would leave own, opponent, and two liberty-level channels per layer. The catch is that SA detection for adjacent own chains would then lack the three-liberty information it needs.

I'm checking whether dropping the ≥3 witness would break self-atari detection, since a healthy chain with three or more liberties could be misread as having only two once a point next to it is filled, which would wrongly flag necessary moves like connecting or filling dame next to our own stones.

I'm considering whether a pointer-doubling scheme could give the log-depth reach that connectivity needs, since each layer can't cover the distance on its own.

I'm working out how to get enough propagation reach, since the chain distance exceeds the layer count. One idea is a witness built from single-stone checks that leaves a few false ataris, and I'm sketching how a stone-level term might stand in for the pair witness.

I'm considering whether to fold the single-stone case into the same per-edge pair unit, so that a stone with no same-coloured neighbour is still covered by a witness of its own.

I'm settling on a cheaper pair of single-stone witnesses per point: one flagging a stone with at least two liberties, and a weaker one flagging at least one liberty. Layer 2 would then infer the pair case by checking whether a same-coloured neighbour has a liberty.

I'm checking whether a non-linear term at the end, counting how many nearby stones of the same colour have a liberty, can be folded into the unit without breaking the count.

I'm now trying to get a binary value out of a sum that can reach two, by subtracting one threshold from another. If nothing nearby has a liberty, the count comes out to two, which isn't binary, so I'm checking whether two units per point and colour can separate those cases cleanly.

I'm laying out a layer budget where the first layer holds own, opponent, and two witness channels per colour, leaving the second layer to clip the propagated values to binary with an extra unit per point. That fills all 486 units in each layer, which leaves the remaining propagation steps with only the liberty-level channels and a small spare budget.

I'm exploring whether a chain-size signal could help with reach: since a large chain usually keeps other liberties, its size might settle the false-atari case without needing extra propagation steps.

I'm now thinking about using how many same-coloured stones sit nearby as a rough size signal: a big chain flags many stones, so the capture bonus could be scaled down when the flagged group is large.

I'm checking whether a "boundary" flag could show that a chain runs past the reach of the witnesses, since the propagation itself can't see that far.

I'm reconsidering whether a simpler practical fix would work, since the false atari mostly comes from big chains. Then I'm checking how the group in game 40 is actually laid out: it has only two liberties, (2,0) and (8,0), and both sit far from the point (7,0).

I'm weighing a shallower alternative: running an extra propagation step for the opponent flags only, and shrinking the tent units from three per point to two. Computing the tents earlier would mean carrying them through every layer, which costs too much, so I'm checking whether a two-unit tent can approximate the same function.

I'm trying a cheaper bump for the tent units, using ramps or a gated relu form, but a simple two-relu version doesn't give a clean bump, so I'm checking whether gating can recover the same shape with two units.

I'm testing whether a two-unit version of the tent, relu(1+x) minus 2·relu(x), can stand in for the three-unit bump. For strongly owned points it goes negative, which would discourage playing inside our own territory, and that looks useful. For opponent-owned points it gives zero. I'm weighing whether that asymmetry is acceptable near our walls, since our boundary moves there.

I'm dropping the idea of a cheap reach fix via more propagation layers, since it only adds one step at a time. The false-atari problem seems to come mostly from big chains, so I'm questioning whether those capture heuristics are worth fixing at all. Before going further, I want to check whether the losses come from strategy instead, so I'm pulling up one of the lost games to see where it went wrong.

