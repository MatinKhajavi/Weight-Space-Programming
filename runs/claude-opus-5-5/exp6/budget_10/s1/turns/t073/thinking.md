I'm checking the liberties of our big bottom group around ply 97: row 8 has only one empty point, (8,4), so the group is down to a single liberty and in atari. Looking back, we had already filled our own eye points at plies 93, 95 and 97, which is what cost us the liberties.

I'm now checking why we filled our own eye points at (8,7) and (8,1). The move at (8,7) scored zero net penalty, probably because a false "in atari" flag on our big chain gave the escape bonus, while the support count came out as zero, so the self-fill looked legal.

I'm weighing a fix for the eye-gating problem: the eye penalty should only be lifted when the chain is actually in atari, since false positives on big chains are what caused this. Restricting that gating to small own chains looks like the safer option, though we lack an isolated-stone unit to check that cleanly.

I'm looking at how to tell an isolated own stone in atari from a big chain, since the NPo flag misfires on big chains. One option is to gate the eye penalty only on own stones with exactly one liberty and no own neighbors, which would avoid the false positives.

I'm working out how to make the eye penalty depend on each neighboring stone's liberties, without a per-stone rule the unit can't express. One idea is to subtract a count of opponent stones around each adjacent own stone, since an isolated stone in atari has three opponent neighbors. The catch is that a stone inside a larger chain would get a smaller subtraction, which is the behavior I want, but the relu still has to make that work as a single linear unit.

I'm working out the normalization for the eye gating: weighting each opponent neighbor of an adjacent own stone by one over that stone's other neighbors, so a fully surrounded isolated stone in atari subtracts exactly one and zeroes the eye penalty. A stone in a two-stone chain would only subtract a partial amount, which leaves a reduced penalty.

I'm now weighing the escape bonus: false own-atari flags on big chains let us "escape" into our own territory, so shrinking it from 9 to 5 might help, since real escapes of big chains are rare. The disaster came from three things combined: the escape bonus, the eye gating, and the A=0 exemption.

I'm now checking the eye-gating fix at (8,7): one neighbor, (7,7), touches a single opponent stone at (6,7), so the weighted count comes out to about 0.33. That leaves the eye penalty at roughly two-thirds strength, which still blocks the self-fill.

I'm now looking at the SA2 exemption for A=0: it's only meant to be safe when a move is a true capture, but when a point has own neighbors, their liberties may be falsely flagged. I'm considering restricting the exemption to points with no own neighbors, so the unit would subtract a term for each own neighbor.

I'm now checking the SA2 exemption for A=0: I'd make it subtract one for each own neighbor, so any point touching our own stones stays penalized unless it's a true capture.

I'm refining the eye-gating term so each neighboring own stone is weighted by how many opponent stones surround it, replacing the NPo flag that misfires on big chains. I'm also changing the SA2 exemption to subtract a count of own neighbors, and I'm considering whether to keep the escape bonus at 9 or lower it, since false flags on big-chain boundaries could give it to points that shouldn't get it.

I'm now asking whether the escape bonus could be limited to small own chains, since lowering it everywhere would also weaken real escapes. The worry is that false flags on big chains would reward filling our own territory and cut into our eye space.

I'm trying to get a stable escape bonus for own chains, since a linear term would penalize every move next to our stones. The bonus should favor small chains in atari without rewarding false flags on big ones, and I'm checking whether the libs in game 48's bottom group could still reveal that difference.

I'm checking whether the bottom group's libs at (8,1) and (8,4) belong to different parity classes, and whether stones near each one are within reach of each other. The neighbors at (8,2) and (8,3) are adjacent, so both parity classes are covered, which would argue the group should be safe. I still need to confirm what the board looked like at ply 93.

I'm checking whether (8,7) counts as an eye: its neighbors are (7,7), (8,6), and (8,8), all ours, so the move looks like a self-fill that the eye penalty should catch. Since the escape bonus doesn't explain it here, I'm looking for another term, such as a capture or the SA2 exemption, that would give it a high value.

Now I'm checking whether the (8,6) point has a same-colored neighbor that could also be an eye. Its surrounding stones and the nearby liberties at (8,5) would tell me whether the chain there was really flagged as unsafe.

I'm checking whether (8,8) is flagged: its nearest lib-bearing stone of the opposite parity, (7,4), sits five steps away, beyond the reach of four, so the eye gating would subtract its flag and weaken the penalty at (8,7).

I'm checking whether (8,7)'s eye-ness is lost because the eye gating subtracts (8,8)'s flag, since (8,8) is out of reach of any lib-bearing stone of the other parity. That would leave the escape bonus at (8,7) as the only thing pulling the move up, with the self-fill penalty suppressed. I'll stick with the local eye gating, which should keep (8,7) blocked, and I'm also noting that the escape bonus can misfire on big chains and push us to fill our own territory.

I'm settling on a sharpening change alongside the eye-gating fix: moving the scale from 2 to 3, since the tactical fix is clearly good and a moderate sharpening looks likely to help as well.

I'm settling on a scale of 3.5 for the sharpening, since the tactical eye-gating fix is already clearly good and a moderate increase should cut the tail of unlikely moves without making play too rigid.

