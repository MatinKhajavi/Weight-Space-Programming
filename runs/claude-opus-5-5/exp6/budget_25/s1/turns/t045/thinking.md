I'm checking the game at index 1, where the margin reads 14.5, and I'm not sure how that squares with a loss, since I had filtered for results below zero. The earlier sorted listing had losses with margins of 13.5 and 14.5, so a positive margin on a loss may just be how those entries are recorded.

I'm checking whether a positive margin on a loss makes sense, and it seems the two losses that look that way ended by repetition, which would explain it. Looking at the opening, the first move at (6,6) isn't what I'd expect from center bonus, so the sampling and loose prior are probably making the early play too random. The main question is what made v4 worse than v3b: v4 captured far less and lost more, which points to the ATT removal as the likely cause.

I'm working through the channel budget, which is five channels: AO, LO, AX, AXS, and LX. One idea is to fold AX and AXS into a single unit, using a shifted relu over the opponent value, so the small-capture cases stay distinguishable at the next layer.

I'm working through how to carry opponent presence through the layers without spending another channel, since VX already marks every opponent stone at this depth. I'm trying to see whether one unit can encode both that presence and the atari status.

I'm weighing whether to drop the LO channel and go back to the nonat-style SELF term, since v3b did better with it than v4 does, and I'm checking how that trades against the channel budget.

I'm weighing whether v3b's SELF term, which skipped the own-liberty count, would be a cheaper fit, since it helped that version even though it also drove up repetition losses. I'm also considering whether some of the L6 features, like EYE, INT, and the LIN patterns, could be computed at L5 and folded in, so I wouldn't have to carry own and opp for them.

I'm working out whether the opponent stones can be carried as a per-point count of empty neighbors instead of as their own channel, so that the own-stone features at the next layer can be computed without another carry.

I'm weighing whether the L6 features can be computed directly there instead of folded into L5, since carrying extra per-point units at that layer would push the total past the budget.

I'm trying to merge the AO and LO channels into one graded value, using a weighted sum of the S-seed and T-seed walk counts, so that atari, two-liberty, and three-or-more-liberty cases come out as three distinct levels.

I'm working out whether a single graded value can separate the atari, two-liberty, and three-plus-liberty cases, with a large multiplier so the three-liberty level sits clearly above the others. From there I'm trying to derive a threshold unit that flags the three-liberty case for SELF without needing a separate own3 channel.

I'm checking whether SAVE can be kept from firing on a large own3 count, since a clamped value would avoid letting one term override the savA requirement.

I'm trying to see whether the merged channel can produce the atari flag directly, since AO would then fall out of G and I wouldn't need a separate unit for it.

I'm weighing whether dropping the AXS channel and letting AX carry an exemption from the SELF penalty would keep the capture cases accurate, since the self-atari false positives drove the repetition losses in v4.

I'm weighing whether dropping the AXS channel and letting AX carry the exemption would work, but a single linear penalty on adjacent AX stones looks too blunt, since real captures often have one or two adjacent stones too. The false-atari repetition cases are the ones where the opponent group has two liberties and we fill one, so LX can't separate them from true captures. I'm trying to see whether keeping both the exemption and the ATT and LO features would need five stone channels at L5, and whether some other unit could be freed up to make room.

I'm looking at whether AX and LX can merge into one graded unit, since atari implies two or fewer liberties, so the three levels of atari, exactly two liberties, and three or more could fit in a single channel.

I'm checking whether a single graded unit can cover atari, exactly-two-liberty, and three-or-more cases with one value, using a staircase where the two-liberty level sits in the middle. A plain relu can't produce that three-level step unless the propagated values are bounded, which they aren't. I'm also wondering how the capture count uses that unit, since it decides whether the merge is worth it.

I'm wondering whether the AX and AXS channels can be replaced by one graded unit, where the size of the opponent group sets how strongly it fires. The catch is that a linear SELF penalty can't act as a threshold, so the exemption would still need a separate way to tell true captures from false-atari repetition cases.

I'm considering whether the false big-atari cases could be handled by strengthening the S-seed reach, since the L2–L4 layers have spare channel capacity, and maybe adding a TX channel to carry the liberty information there.

I'm considering whether the opponent stones' presence and their liberty status can share one unit by encoding opp(q) through graded values, so the L6 features that need e(p) and the line patterns could still be computed without carrying opp separately.

I'm considering whether the opponent stones could be carried as a single graded unit, but a linear combination of relu outputs doesn't collapse into one clean value. Then I'm wondering if e(p), the count of empty neighbors, could be computed at L5 as its own unit, which would let L6 drop the opponent carry altogether.

I'm considering whether the own and opponent stone units can be merged further, since a point is only ever one color. A single per-point unit might carry the own-stone status when the point is own and the opponent-stone status when it is opp, summing cleanly because the other term is zero.

I'm considering whether a single unit could carry both the own-stone and opponent-stone status, since AO and AX are never active on the same point. Their sum is a relu of own plus opp minus the S-walk term and the scaled X-walk term, which reduces to AO on own points and AX on opponent points. The catch is that downstream I need AO and AX separately, so I'm weighing how to split them back apart without a second unit.

I'm checking whether one unit can hold both the own-stone atari status and the opponent-stone atari status, since the two never coincide on a point. Then I'm asking whether the head actually needs all of these status channels, given that the capture bonus only needs the opponent atari count, SELF needs the empty count and own three-liberty information, and SAVE needs only the own atari status.

I'm checking whether ATT could drop the LX channel and still work, perhaps by testing whether a move leaves an adjacent opponent group in atari, but there's no obvious way to do that without the two-liberty information. I'm now turning to the earlier v3b results to see how much the ATT and nonat-SELF terms actually contributed.

I'm sizing up how many channels the stone-status features need at layer 5, with the opponent counts running about 57% against 38% for the earlier variant. The next idea is to store the opponent stones' liberty status in a third graded unit, so the layer can hold the atari, two-liberty and three-plus-liberty cases without separate carries.

I'm noticing that the opponent stones could be split into an atari-or-two-liberty unit plus a three-or-more-liberty unit, so their presence is recoverable as their sum without carrying the opponent stones separately. I'm checking whether that third unit can be computed with a single relu, since a min of the opponent presence and the T-seed walk is not directly expressible.

I'm checking whether the opponent-stone unit can be recovered without a carry: if the liberty-count units are unbounded, OX3 can't be cut down to a clean indicator with a single relu, so I'd need to see whether the downstream uses of opp(q), like the empty-neighbor count, still work when OX3 is larger than one.

I'm checking whether one unit can carry the opponent-stone presence along with its liberty status, since the head only needs that combined value, but unbounded propagated values would make the empty-neighbor count go strongly negative near large opponent groups. I'm looking at whether the propagation can be clamped cheaply, perhaps at the final layer, so the units stay bounded.

I'm wondering whether mixing the own and opponent status units into one shared channel would save capacity, but then the SAVE sum over neighbors would pick up adjacent opponent atari stones and fire when it shouldn't, which is exactly the capture case I'm trying to keep separate.

I'm considering merging the own-atari and opponent-atari statuses into one "atari-anything" unit, since they never sit on the same point. That would let the capture bonus and the save term share a single channel, though it would also reward extending a dead-end own stone, which the SELF penalty would need to counter. Then I'd set the SAVE unit up separately from this merged signal.

I'm considering merging the own-stone and opponent-stone liberty statuses into a second shared unit, which would let the own-stone liberty count be recovered from the same channel. I'm checking whether the own-three-liberty information can be derived that way without a separate carry.

I'm considering whether the own-atari and opponent-atari units could merge into one "atari-anything" unit, since they never sit on the same point, and whether the opponent-two-liberty term could be expressed as the merged two-liberty unit minus that atari unit. That would make the defend-or-attack term fire near own two-liberty groups as well as opponent ones.

I'm checking whether the SELF term still holds when p sits next to opponent two-liberty stones, since merging the two-liberty units makes the opponent's stones push the penalty up. Adding an opponent-stone compensation into the own-three term might cancel that, but I'm not sure yet whether it does.

I'm checking whether the opponent contribution can be patched back in with an opp term, but adding raw opponent presence to the SELF sum flips the sign for strong opponent groups, so it would wrongly lower the penalty next to them.

I'm checking whether one shared unit for own and opponent atari can stand in for both, since the two never coincide on a point, and then whether the own-two-liberty count can be recovered from the same channel.

I'm checking whether one shared unit for own and opponent atari can stand in for both, since the two never coincide on a point. With that merge, the capture-or-save term and the attack term can both be written from the same channel, with the separate atari information subtracted back out where it's needed.

I'm settling on a layer-5 layout of four units: own, opp, a merged atari unit, and a shared status unit. The merged atari unit covers both colors' atari stones, which works because they never share a point, and the remaining terms can be rewritten from it with a correction for the other color.

I'm settling the final layer's layout: the L5 units come to 486 in total, and the U1 status unit is computed from own and opponent presence minus the save and void corrections. The SELF exemption still relies on the small-group unit, while larger captures earn their bonus through U1 without an exemption, as before. Next I need to fix what the L6 terms are.

Now I'm settling the trunk and head layout: I'll drop the CONST and COLOR carries, since the biases already handle the gating, and keep only own and opp. For the bonuses, I'm setting the capture term around 6 and the save weight near 8, so a save of an own atari stone still scores strongly.

