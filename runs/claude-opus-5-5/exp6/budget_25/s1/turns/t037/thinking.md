I'm checking whether playing at (8,3) for Black is safe: the (8,4) stone would have only one liberty left after the move, and I still need to see whether the neighboring white stones can be captured.

I'm checking the white group's liberties: after my move it drops to two, and I'm weighing whether the apparent capture is real or a false read on a big group. The self-atari cost looks underweighted, which is why the evaluation keeps landing near zero, and I'm considering how to strengthen it and shrink the channel budget to fit.

Still working out the layer budget: dropping the color carry saves only one channel, and the own/opp carries are still needed for the later layers.

I'm weighing whether to drop the LX (attack) channel so the layer fits the budget: that would leave AO, AX, AXS and LO, four channels that fit. I'm also considering whether the capture bonus can use the graded AXS values instead of AX, with a weaker separate bonus for larger captures.

I'm weighing whether to drop the ATT channel, since our captures may depend less on it than I assumed. Against that, strong opponents' territory fights are decided by tactics, so I'm not sure ATT is expendable. I'm also considering whether to compute LX without LO and detect the connect-to-own-2-lib case some other way.

I'm looking at whether to replace the own-liberty channel with a per-point unit so the layer fits the budget, since the later layers still need own and opponent occupancy for the adjacent and distance-two counts.

I'm settling on the channel budget: I'd drop the ATT channel and rely on the L6 room that frees up, though I'm still weighing whether tactical play in strong territory fights depends on it.

I'm weighing which channel to drop, since five channels at this layer don't fit the budget: either LX or AXS has to go. I'm also looking at whether the small-size condition can be folded into AX with a threshold instead.

I'm considering whether the capture-count channel can be built from the opponent's TX values by adding a mass term, so that a "small and no-stones-of-that-kind" signal comes out of a single unit. The catch is that this only gives "small with at most two liberties," not true atari, so I'd still need an AND with the atari signal inside the self-atari unit, and that AND costs a unit per point.

I'm trying a linear shortcut: summing over neighbouring stones a term that is high only when a stone is both in atari and small, with opponent stones gated in. That sum counts stones that qualify minus stones that qualify on neither count, so a mixed neighbourhood can cancel out and the exemption would fail, which I'd accept as an approximation. I'm starting to write the self-atari unit with that exemption term.

I'm testing a different way to get the small-group size into the self-atari exemption: instead of a separate channel, the graded size value itself could drive the check, using a much lower cutoff so that only very small groups register.

I'm settling on AXS for both the capture bonus and the SELF exemption, so the L5 budget comes to four channels: AO, LO, AXS and LX. Graded AXS gives strong weight to single stones and dominoes, but a three-stone group only registers at its ends and a group of four or more doesn't register at all, so I'm weighing whether that's acceptable since big groups rarely get captured on purpose.

I'm checking whether a higher cutoff K would catch more capture value without also flagging long chains as small, since K=28 would wrongly mark long chain ends as capturable. I'm leaning toward K=13, which stays exact for groups of up to two stones plus the ends of three-stone lines, and wondering how often real captures of bigger groups actually happen against these bots.

Weighing the capture bonus for groups of three to ten stones, I see that without it we'd miss them, though the contact bonus and ATT still pull moves toward opponent stones. I'm considering a moderate bonus for bigger ataris through LX, but that's inaccurate for large groups, so I'm not sure it's worth it. I'm also wondering whether AX and AXS could both come from the same VX channel, at the cost of LX, and I'm looking at whether ATT is really necessary by checking how often two-liberty groups actually get captured.

I'm working out whether L6 can take over some of the per-point work, since it still needs each point's own/opponent occupancy for the adjacency counts, and I'm checking if the INT and EYE terms can be moved down to L5 from the L4 carries.

I'm checking whether a per-point "own stone not in atari" unit can be built from own occupancy and the seed sum, so the neighbour count doesn't need a separate own-occupancy channel for each point.

I'm settling the L5 budget as four channels: AO, LO, the graded AXS, and AX from VX with the same K=1024 cutoff. The rest of the layer is carries.

I'm weighing whether to drop the ATT channel or the big-capture bonus, since five channels at this layer won't fit. Big-group captures are prone to false positives, but with the own-liberty channel restored, a self-atari move next to a two-liberty group would get penalised, so I'm inclined to keep the big-capture bonus.

I'm dropping the LX and ATT channels, which leaves spare room in the L2–L4 layers and in L6 for something else. I'm also working out how the SELF exemption uses the small-group channel, and how the capture bonus combines the any-size channel with the small-group one, and I'm checking the libs-after estimate that comes from the own-liberty counts.

I'm setting the self-atari term as relu(2 − e − 2·own3 − 3·capS'), so a group with two liberties and no empty neighbours gets penalised, while a three-liberty group doesn't.

I'm checking that a single-stone or one-capture case is covered: an empty neighbourhood with no own or captured stones means the move is suicide and therefore illegal. The harder case is a real big capture where the move has no empty neighbours but would still capture a large group, since the surrounding friendly group might have only two liberties and trigger the self-atari penalty. I'm considering a partial exemption for big captures so that a valuable semeai capture isn't blocked by the penalty.

I'm now checking whether the false-atari cases in those repetition games can be told apart from real captures using the geometry alone, since the propagation misses seeds in long groups.

I'm walking through the white group in that game: row 7 has five stones from column 0 to 4, and row 8 has two more at columns 1 and 2. Its two liberties are at (8,0) and (8,3). Checking the seed at (7,0), it has one empty neighbour and no own neighbour with an empty side, so the seed sum comes out to 2.

I'm checking the seeds at (8,1) and (8,2), which each have one liberty and are adjacent, so they should count as separate threats. That means (8,2) should register as seeded next to (8,3), yet the 0.96 probability on (8,3) still seems off, so I'm wondering why that move was chosen.

I'm settling on a three-channel layout (SO, TO, V) for the rest of the design, and checking that the restored LO term already handles the weak-SELF case.

Next I'm laying out the layer-by-layer channel plan for the trunk: the carries plus the edge and block channels in L1, then the own/opponent stone channels and the VX channel carried through L2–L4, and at L5 the AO, LO and AX channels with the AX cutoff at K=1024.

Looking at the trunk budget again, the L5 layer has only 24 spare units, and L2–L4 each have about 105 spare. I'm considering whether a global feature like a score estimate could fit into that spare room, but it would be linear and wouldn't need it. I'm still weighing what the leftover six-layer capacity could usefully hold, perhaps a contestedness measure for influence.

I'm now considering whether a "contested region" feature can go into the L6 units: a smoothed minimum of own and opponent influence, built from the linear T and I terms plus a negative-part unit n(x) for the output layer.

I'm setting the VX channel's bias so that an opponent stone at q contributes exactly the right amount, with the weight on the opponent-occupancy input chosen as a power of two to keep it exact in bf16.

Setting the VX bias so an opponent stone's contribution comes out exact, with weights chosen as powers of two to stay exact in bf16.

I'm checking the L5 terms now: AX comes out as relu of the opponent occupancy minus the mass divided by 1024, so an empty point sits just under 1 when nothing nearby is seeded. AXS is the graded version, relu(13·opp minus the mass). I'm also confirming the mass values: a single stone gives 1, a domino gives 8, which matches the table.

I'm checking whether a self-atari gate can stop the SAVE unit from firing without an atari, by raising the threshold so that only a true extension of a one-liberty group passes.

I'm settling the self-atari term as relu(2 − e − 2·own3 − 3·ΣAXS), with no exemption for big captures, since an exemption would let the false-atari cases through again.

I'm settling the weights for the self-atari and capture terms: the self-atari weight at -10, and the capture bonus at 6 for any atari and 0.4 for single-stone captures. I'm also deciding to keep the pattern-weight shape from the earlier design rather than reverting, since the drop in win rate came from repetition losses, not from the shape itself.

